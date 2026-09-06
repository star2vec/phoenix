"""Training driver — single-device port of vendor run.py.

Reuses the paper's own modules unchanged (vendor/reasoning-by-superposition:
STokenizer, Coconut, dataset builders, MyCollator, config JSON). Replaces only
the orchestration: no torchrun/FSDP/wandb; adds per-stage early stopping,
JSONL metrics, resumable checkpoints,
and a wall-clock cap.

Faithfulness notes:
- Effective batch 256 = paper's batch_size_training 128 x 2 GPUs, realized here
  as 128 x grad-accum 2.
- Stage schedule identical: scheduled_stage = epoch // 25, epochs 0..299;
  uniform_prob 0.1 stage mixing lives inside the vendor dataset builder.
- Early stopping only trims epochs within a stage (patience on val accuracy);
  it never changes the data, loss, optimizer, or schedule otherwise.
- get_graph_latent_question_dataset is reimplemented here because the vendor
  version only builds its dataset in the multi-GPU branch (NameError on one
  device); logic is copied verbatim from the single-device branches of the
  sibling builders.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "reasoning-by-superposition"
sys.path.insert(0, str(VENDOR))

import torch
import torch.optim as optim
from transformers import AutoConfig, AutoModelForCausalLM

from dataset import MyCollator, expand_data, get_graph_latent_cot_dataset
from utils import Config, set_seed

from fast_coconut import FastCoconut, FastSTokenizer


def build_stage_question_dataset(dataset_path, scheduled_stage, tokenizer):
    """Stage-matched validation questions.

    For each sample, k = min(scheduled_stage, len(steps)) + 1 mirrors the
    training objective at this stage (vendor expand_data): k <= len(steps)
    means "predict a depth-k valid-path node" (correct iff the generated token
    is in neighbor_k[k]); k == len(steps)+1 is the full task (correct iff it
    equals the target). Early stages therefore get an accuracy signal that
    tracks what they are actually being taught — full-task accuracy is ~0
    there no matter how well the stage is learning.
    """
    base_dataset = json.load(open(dataset_path))
    processed = []
    for idx, sample in enumerate(base_dataset):
        max_steps = len(sample["steps"])
        k = min(scheduled_stage, max_steps) + 1
        question, _ = expand_data(sample, k, max_steps, neg_sampling=False)
        question_tokenized = tokenizer.encode(question, add_special_tokens=False)
        expected = (
            {int(sample["target"])}
            if k == max_steps + 1
            else {int(n) for n in sample["neighbor_k"][str(k)]}
        )
        processed.append(
            {
                "input_ids": question_tokenized,
                "attention_mask": [1] * len(question_tokenized),
                "position_ids": list(range(len(question_tokenized))),
                "idx": idx,
                "expected": expected,
            }
        )
    return processed


@torch.no_grad()
def eval_generation(model, dataset, collator, device):
    """Greedy 1-token generation; node tokens "0".."30" have id == int(node)."""
    model.eval()
    cor, total = 0, 0
    for feature in dataset:
        expected = feature["expected"]
        batch = collator([{k: v for k, v in feature.items() if k != "expected"}])
        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if v is not None and k not in ["idx", "position_ids"]
        }
        outputs = model.generate(**batch, max_new_tokens=1)
        cor += outputs[0, -1].item() in expected
        total += 1
    return cor / total, cor, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None, help="default: seed{seed}")
    p.add_argument("--device", default="mps")
    p.add_argument("--num-epochs", type=int, default=300)
    p.add_argument("--epochs-per-stage", type=int, default=25)
    # 64 x 4 = effective 256 = the paper's 128 x 2 GPUs; batch 128 stalls the
    # MPS allocator on 16 GB (26 s/step vs 2.3 s/step at 64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--uniform-prob", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3, help="per-stage early stop (stages 0-3)")
    # stages >= 4 are one continuous full-task phase (the paper spends 175+
    # epochs there); per-stage patience-3 starved it.
    p.add_argument("--full-task-patience", type=int, default=10)
    p.add_argument("--min-full-epochs", type=int, default=40)
    # periodic checkpoints for the training-trajectory analysis: every
    # curriculum-stage end, the first full-task epoch, and every N full-task
    # epochs after that (0 disables; best.pt and latest_state.pt are unaffected)
    p.add_argument("--save-every", type=int, default=10)
    # Amendment 1: train on a relabeled copy of the data (same file names)
    p.add_argument("--data-dir", default=str(VENDOR / "data"),
                   help="directory holding prosqa_{train,valid}_graph_4_coconut.json")
    # run stages 0-3 on the paper's fixed 25-epoch schedule (no early
    # stopping); with patience-3 an early run's readout was broken at exactly
    # the depths stages 1-2 teach
    p.add_argument("--fixed-early-stages", action="store_true")
    p.add_argument("--wall-clock-cap-hours", type=float, default=36.0)
    p.add_argument("--tiny-run", action="store_true", help="tiny run to check pipeline")
    args = p.parse_args()

    run_name = args.run_name or f"seed{args.seed}"
    ckpt_dir = ROOT / "ckpts" / run_name  # run_name may contain a subdirectory, e.g. seed0/relabel
    results_dir = ROOT / "results" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.jsonl"
    state_path = ckpt_dir / "latest_state.pt"

    device = torch.device(args.device)
    set_seed(args.seed)

    tokenizer = FastSTokenizer()
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    base = AutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained(str(VENDOR / "configs/symbol-2layer-8head-768dim.json"))
    )
    model = FastCoconut(base, latent_id, start_id, end_id, tokenizer.eos_token_id).to(
        device
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # dataset builders read these fields from a Config object
    configs = Config(
        {
            "debug": False,
            "uniform_prob": args.uniform_prob,
            "train_path": str(Path(args.data_dir) / "prosqa_train_graph_4_coconut.json"),
            "val_path": str(Path(args.data_dir) / "prosqa_valid_graph_4_coconut.json"),
        }
    )

    start_epoch = 0
    best_val_acc = -1.0
    if state_path.exists():
        # cpu map: RNG state must stay a CPU ByteTensor; load_state_dict moves
        # model/optimizer tensors onto the module's device itself
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_val_acc = state.get("best_val_acc", -1.0)
        # restore RNG so per-epoch dataset regeneration continues deterministically
        if "py_random" in state:
            import random as _r

            _r.setstate(state["py_random"])
            torch.set_rng_state(state["torch_rng"].cpu())
        print(f"[resume] from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    stage_question_cache = {}

    def stage_questions(stage):
        if stage not in stage_question_cache:
            ds = build_stage_question_dataset(configs.val_path, stage, tokenizer)
            if args.tiny_run:
                ds = ds[:16]
            stage_question_cache[stage] = ds
        return stage_question_cache[stage]

    t_start = time.time()
    stage_best = {}  # stage -> best val acc seen in that stage
    stage_stale = {}  # stage -> epochs since improvement
    epoch = start_epoch

    # full-task phase counters, replayed from the metrics log (stateless)
    full_epochs, full_best, full_stale = 0, -1.0, 0
    if metrics_path.exists():
        for line in open(metrics_path):
            r = json.loads(line)
            if r.get("full_task"):
                full_epochs += 1
                if r["stage_val_acc"] > full_best:
                    full_best, full_stale = r["stage_val_acc"], 0
                else:
                    full_stale += 1

    while epoch < args.num_epochs:
        scheduled_stage = epoch // args.epochs_per_stage
        ep_t0 = time.time()

        train_dataset = get_graph_latent_cot_dataset(
            configs.train_path, scheduled_stage, configs, tokenizer
        )
        if args.tiny_run:
            train_dataset = train_dataset[: args.batch_size * 4]
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=collator,
            shuffle=False,  # vendor pre-shuffles inside the builder each epoch
        )

        model.train()
        total_loss, n_steps = 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: batch[k].to(device) for k in batch if k != "idx"}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            total_loss += outputs.loss.item()
            n_steps += 1
            if (step + 1) % args.grad_accum == 0 or step == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
            if device.type == "mps":
                # drain the command queue per micro-step: unbounded queuing keeps
                # buffers alive and stalls the allocator (measured 10x slowdown)
                torch.mps.synchronize()

        val_acc, cor, total = eval_generation(
            model, stage_questions(scheduled_stage), collator, device
        )
        # from stage 4 on, every sample's stage question IS the full task
        # (max hops = 4), so val_acc doubles as full-task accuracy there
        is_full_task = scheduled_stage >= 4

        elapsed_h = (time.time() - t_start) / 3600
        rec = {
            "epoch": epoch,
            "stage": scheduled_stage,
            "train_loss": round(total_loss / max(n_steps, 1), 6),
            "stage_val_acc": round(val_acc, 6),
            "full_task": is_full_task,
            "val_cor": cor,
            "val_total": total,
            "epoch_seconds": round(time.time() - ep_t0, 1),
            "elapsed_hours": round(elapsed_h, 3),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"[epoch {epoch}] stage {scheduled_stage} "
            f"loss {rec['train_loss']:.4f} stage_val_acc {val_acc:.4f}"
            f"{' (=full-task)' if is_full_task else ''} "
            f"({rec['epoch_seconds']}s, {elapsed_h:.2f}h total)"
        )

        if is_full_task and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            with open(results_dir / "best.json", "w") as f:
                json.dump({"epoch": epoch, "val_acc": val_acc}, f)

        import random as _r

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "py_random": _r.getstate(),
                "torch_rng": torch.get_rng_state(),
            },
            state_path,
        )

        if args.save_every > 0:
            stage_end = (not is_full_task) and (epoch + 1) % args.epochs_per_stage == 0
            periodic = is_full_task and (full_epochs == 0 or (full_epochs + 1) % args.save_every == 0)
            if stage_end or periodic:
                torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")

        if is_full_task:
            # single continuous full-task phase; stop only on
            # a long plateau after a minimum number of full-task epochs
            full_epochs += 1
            if val_acc > full_best:
                full_best, full_stale = val_acc, 0
            else:
                full_stale += 1
            if full_epochs >= args.min_full_epochs and full_stale >= args.full_task_patience:
                print(
                    f"[early-stop] full-task phase plateaued (patience "
                    f"{args.full_task_patience}, {full_epochs} full-task epochs, "
                    f"best {full_best:.4f}); ending training"
                )
                break
            epoch += 1
        elif args.fixed_early_stages:
            epoch += 1  # paper's schedule: full 25 epochs per curriculum stage
        else:
            # per-stage early stopping for curriculum stages 0-3
            sb = stage_best.get(scheduled_stage, -1.0)
            if val_acc > sb:
                stage_best[scheduled_stage] = val_acc
                stage_stale[scheduled_stage] = 0
            else:
                stage_stale[scheduled_stage] = stage_stale.get(scheduled_stage, 0) + 1

            if stage_stale[scheduled_stage] >= args.patience:
                next_boundary = (scheduled_stage + 1) * args.epochs_per_stage
                print(
                    f"[early-stop] stage {scheduled_stage} plateaued "
                    f"(patience {args.patience}); jumping epoch {epoch} -> {next_boundary}"
                )
                epoch = next_boundary
            else:
                epoch += 1

        if elapsed_h > args.wall_clock_cap_hours:
            print(f"[cap] wall clock {elapsed_h:.1f}h > {args.wall_clock_cap_hours}h; stopping.")
            break
        if args.tiny_run and epoch >= start_epoch + 2:
            print("[tiny-run] pipeline OK, exiting.")
            break

    print(f"[done] best val acc {best_val_acc:.4f}; metrics: {metrics_path}")


if __name__ == "__main__":
    main()
