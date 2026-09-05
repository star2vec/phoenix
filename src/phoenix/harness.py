"""Model harness: load a checkpoint, run one prompt, edit a thought, read T.

Runner wraps a trained FastCoconut. `measure` renders a sample with the vendor
prompt builder, runs one forward, and returns

  T      = softmax(p)[target] / (softmax(p)[target] + softmax(p)[decoy])
           at the answer position, and
  escape = 1 - (softmax(p)[target] + softmax(p)[decoy]).

An optional edit is applied to the recycled thought at one step. All edit
operations rescale their output to the PRE-EDIT thought norm ||t|| (measured
thought norms are ~27, not 1), so an edited thought stays on the norm shell
the model actually produces:

  SUBTRACT v: t' = ||t|| * normalize(t - <t,u^_v> u^_v)
  COLLAPSE v: t' = ||t|| * sign(<t,u^_v>) u^_v
  INJECT  x: t' = ||t|| * normalize(t + g*u^_x), g = median <t,u^_f> over frontier f
  Controls: same operation with a random unit direction and matched magnitude.

The per-node direction u^_v is the input embedding of node token v by default,
or a row of a caller-supplied basis (e.g. probe directions from fit_probes.py).
"""

import random
import statistics
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "reasoning-by-superposition"
sys.path.insert(0, str(VENDOR))

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from dataset import expand_data
from fast_coconut import FastCoconut, FastSTokenizer


def bfs_depths(edges, root):
    adj = {}
    for s, t in edges:
        adj.setdefault(s, []).append(t)
    depth = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth


class Runner:
    def __init__(self, checkpoint, device="mps", seed=0, basis_vectors=None):
        """basis_vectors: optional (vocab, 768) tensor replacing input
        embeddings as the per-node basis (e.g. probe directions); zero rows
        mean 'no direction for this node' and are an error to use."""
        self.device = torch.device(device)
        self.tok = FastSTokenizer()
        self.latent_id = self.tok.convert_tokens_to_ids("<|latent|>")
        base = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(
                str(VENDOR / "configs/symbol-2layer-8head-768dim.json")
            )
        )
        self.model = FastCoconut(
            base,
            self.latent_id,
            self.tok.convert_tokens_to_ids("<|start-latent|>"),
            self.tok.convert_tokens_to_ids("<|end-latent|>"),
            self.tok.eos_token_id,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.wte = base.transformer.wte.weight  # (40, 768)
        self.basis_vectors = (
            None if basis_vectors is None else basis_vectors.to(self.device)
        )
        self.rng = random.Random(seed)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def u_hat(self, node):
        if self.basis_vectors is not None:
            v = self.basis_vectors[node].detach()
            if v.norm() == 0:
                raise ValueError(f"no probe direction for node {node}")
        else:
            v = self.wte[node].detach()
        return v / v.norm()

    def rand_unit(self):
        v = torch.randn(self.wte.shape[1], generator=self.gen)
        return (v / v.norm()).to(self.device)

    @torch.no_grad()
    def measure(self, sample, edit_step=None, edit=None):
        """One forward; returns (T, escape). edit: callable(vec, info)->vec
        applied to the recycled thought at thought step edit_step (1-based)."""
        max_steps = len(sample["steps"])
        question, _ = expand_data(sample, max_steps + 1, max_steps, neg_sampling=False)
        ids = self.tok.encode(question)
        input_ids = torch.tensor([ids], device=self.device)
        n = len(ids)
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
            "position_ids": torch.arange(n, device=self.device).reshape(1, -1),
        }

        info = {}

        def thought_edit(pass_idx, vec):
            if edit is None or pass_idx != edit_step - 1:
                return vec
            return edit(vec, info)

        out = self.model(
            **batch, thought_edit=None if edit is None else thought_edit
        )
        p = F.softmax(out.logits[0, -1], dim=-1)
        pt, pd = p[sample["target"]].item(), p[sample["neg_target"]].item()
        T = pt / (pt + pd) if (pt + pd) > 0 else 0.5
        escape = 1.0 - (pt + pd)
        return T, escape

    # --- edit operations (all rescale to the pre-edit thought norm) ----------
    def op_subtract(self, node):
        u = self.u_hat(node)
        def f(t, info):
            t2 = t - (t @ u) * u
            return t2 / t2.norm() * t.norm()
        return f

    def op_subtract_random_matched(self, node):
        u = self.u_hat(node)
        r = self.rand_unit()
        def f(t, info):
            t2 = t - (t @ u) * r  # same coefficient magnitude, random direction
            return t2 / t2.norm() * t.norm()
        return f

    def op_collapse(self, node):
        u = self.u_hat(node)
        def f(t, info):
            return torch.sign(t @ u) * u * t.norm()
        return f

    def op_collapse_random(self):
        r = self.rand_unit()
        def f(t, info):
            return r * t.norm()
        return f

    def op_inject(self, node, frontier):
        u = self.u_hat(node)
        us = [self.u_hat(fnode) for fnode in frontier]
        def f(t, info):
            g = statistics.median(float(t @ uf) for uf in us)
            t2 = t + g * u
            return t2 / t2.norm() * t.norm()
        return f

    def op_inject_random(self, frontier):
        r = self.rand_unit()
        us = [self.u_hat(fnode) for fnode in frontier]
        def f(t, info):
            g = statistics.median(float(t @ uf) for uf in us)
            t2 = t + g * r
            return t2 / t2.norm() * t.norm()
        return f


def answer_branch_root(sample, depth):
    """Depth-1 ancestor of the target along shortest paths (training graphs)."""
    parents = {}
    for s, t in sample["edges"]:
        parents.setdefault(t, []).append(s)
    frontier = {sample["target"]}
    d = depth[sample["target"]]
    while d > 1:
        frontier = {
            p
            for v in frontier
            for p in parents.get(v, [])
            if depth.get(p) == d - 1
        }
        d -= 1
    return sorted(frontier)[0]
