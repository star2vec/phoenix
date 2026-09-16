"""Natural-language ProsQA prompts for the fine-tuned GPT-2 COCONUT models
(experiment 7). The from-scratch symbol path in prompts.py is untouched.

The vendor graphs carry the original ProsQA text. Its format, verified byte
for byte on all 15,204 vendor graphs (tests/test_gpt2_plumbing.py):

    "{Name} is a {y}."        when the source symbol is capitalised (a person)
    "Every {x} is a {y}."     otherwise
    sentences joined by single spaces, then " Is {Root} a {c1} or {c2}?"

The stored `edges` list is in a different order from the sentences, so the
baseline edge order is read off the question. The official COCONUT pipeline
tokenises `question + "\\n"`, then <|start-latent|>, six <|latent|> (always six
at evaluation), <|end-latent|>, then "### {Root} is a {answer}." and eos.

Prompts here end with the common token prefix of the two candidates' answer
strings, so the next token at the last position is where the candidates
diverge (on every recipient the prefix is the fixed frame "### Root is a").

Layout (absolute token positions), from the fast tokenizer's offsets:
  slots[j]          tokens of sentence j as a tuple ordered (source-name first
                    token, target-name first token, period, remaining tokens);
                    position 0 (GPT-2's attention sink) is never in a slot,
                    and a source anchor that would be position 0 is None
  sink              0
  root, c1, c2      first token of the root name and of the two candidates in
                    the question
  start, latents, end, a (last position), n (length)
  thought1_query    the start-latent position: the query that computes
                    thought 1 (the symbol layout's "root" class)
  extra_query_classes  latent0..latent{K-1} and search_latent = latents 0..L-2
                    (the passes the removal acts on)

K is the latent count (6); L is the graph's solution length (3 or 4).
"""

import random
import re
from dataclasses import dataclass

from prompts import Prompt
from sets import recipients, test_pin

START, END, LATENT_TOK = "<|start-latent|>", "<|end-latent|>", "<|latent|>"
K_LATENTS = 6
_SENT = re.compile(r"^(?:Every )?(\w+) is a (\w+)\.$")
_QUESTION = re.compile(r" Is (\w+) a (\w+) or (\w+)\?$")

_TOK = None  # bound once by the GPT-2 runner; layout() and readout() use it


def bind_tokenizer(tok):
    global _TOK
    _TOK = tok


def tokenizer():
    assert _TOK is not None, "nl.bind_tokenizer(tok) first (GPT2Runner does it)"
    return _TOK


def special_ids(tok):
    return {"start": tok.convert_tokens_to_ids(START), "end": tok.convert_tokens_to_ids(END),
            "latent": tok.convert_tokens_to_ids(LATENT_TOK)}


def is_person(name):
    return name[0].isupper()


def sentence(names, s, t):
    return f"{names[s]} is a {names[t]}." if is_person(names[s]) else f"Every {names[s]} is a {names[t]}."


def parse_question(sample):
    """(edge order, displayed candidate order) read off the stored question."""
    names = sample["idx_to_symbol"]
    inv = {n: i for i, n in enumerate(names)}
    q = sample["question"]
    m = _QUESTION.search(q)
    assert m, q[-80:]
    body = q[: m.start()]
    edges = []
    for s in body[:-1].split(". "):
        mm = _SENT.match(s + ".")
        assert mm, s
        edges.append([inv[mm.group(1)], inv[mm.group(2)]])
    assert inv[m.group(1)] == sample["root"]
    cands = (inv[m.group(2)], inv[m.group(3)])
    assert set(cands) == {sample["target"], sample["neg_target"]}
    return edges, cands


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@dataclass
class NLPrompt(Prompt):
    names: list      # idx_to_symbol
    L: int           # solution length (steps); K is the latent count

    @classmethod
    def from_sample(cls, sample, K=K_LATENTS):
        edges, cands = parse_question(sample)
        return cls(edges, cands, sample["root"], K, sample["target"], sample["neg_target"],
                   list(sample["idx_to_symbol"]), len(sample["steps"]))

    # --- text ---------------------------------------------------------------
    def question(self):
        body = " ".join(sentence(self.names, s, t) for s, t in self.edges)
        return body + f" Is {self.names[self.root]} a {self.names[self.cands[0]]} or {self.names[self.cands[1]]}?"

    def answer_text(self, node):
        return f"### {self.names[self.root]} is a {self.names[node]}."

    def text(self, K=None, latent=LATENT_TOK, markers=True):
        """Display form; ids() is the real tokenisation path."""
        K = self.K if K is None else K
        mid = ([START] if markers else []) + [latent] * K + ([END] if markers else [])
        return self.question() + "\n" + "".join(mid) + self.answer_prefix_text()

    def answer_prefix_text(self):
        tok = tokenizer()
        _, _, pre, _ = self.pieces(tok)
        return tok.decode(pre)

    # --- ids ----------------------------------------------------------------
    def pieces(self, tok=None, K=None, markers=True):
        """(question ids, latent block ids, answer-prefix ids, (target answer
        ids, decoy answer ids)) exactly as the official pipeline tokenises."""
        tok = tokenizer() if tok is None else tok
        K = self.K if K is None else K
        q_ids = tok.encode(self.question() + "\n", add_special_tokens=True)
        sp = special_ids(tok)
        lat = ([sp["start"]] if markers else []) + [sp["latent"]] * K + ([sp["end"]] if markers else [])
        a_t = tok.encode(self.answer_text(self.target), add_special_tokens=False)
        a_d = tok.encode(self.answer_text(self.decoy), add_special_tokens=False)
        P = common_prefix_len(a_t, a_d)
        assert P < min(len(a_t), len(a_d))
        return q_ids, lat, a_t[:P], (a_t, a_d)

    def ids(self, tok=None, K=None, markers=True, **kw):
        q, lat, pre, _ = self.pieces(tok, K, markers)
        return q + lat + pre

    def readout(self, tok=None):
        """Token ids read at the last position: target, decoy, the graph's
        concept-name tokens under the same prefix, and collisions."""
        tok = tokenizer() if tok is None else tok
        _, _, pre, (a_t, a_d) = self.pieces(tok)
        P = len(pre)
        out = {"target": a_t[P], "decoy": a_d[P], "prefix_len": P}
        nodes, by_name = set(), {}
        for v in sorted(self.nodes()):
            if is_person(self.names[v]):
                continue
            a_v = tok.encode(self.answer_text(v), add_special_tokens=False)
            if a_v[:P] == a_t[:P] and len(a_v) > P:
                by_name[self.names[v]] = a_v[P]
                nodes.add(a_v[P])
        out["nodes"] = sorted(nodes)
        out["node_by_name"] = by_name
        out["n_collided"] = sum(1 for v, t in by_name.items()
                                if t in (out["target"], out["decoy"])
                                and v not in (self.names[self.target], self.names[self.decoy]))
        return out

    # --- layout -------------------------------------------------------------
    def layout(self, tok=None):
        tok = tokenizer() if tok is None else tok
        q_text = self.question() + "\n"
        enc = tok(q_text, return_offsets_mapping=True, add_special_tokens=True)
        q_ids, offs = enc["input_ids"], enc["offset_mapping"]
        _, lat, pre, _ = self.pieces(tok)
        assert q_ids == tok.encode(q_text, add_special_tokens=True)

        def tok_at(ch):
            """index of the token covering character ch"""
            for i, (o0, o1) in enumerate(offs):
                if o0 <= ch < o1:
                    return i
            # a leading-space token may be trimmed to the word; fall back to the
            # token whose span starts right after ch
            for i, (o0, o1) in enumerate(offs):
                if o0 > ch:
                    return i
            raise KeyError(ch)

        slots, pos = [], 0
        for j, (s, t) in enumerate(self.edges):
            sent = sentence(self.names, s, t)
            if j > 0:
                pos += 1  # the joining space
            src_name, tgt_name = self.names[s], self.names[t]
            src_ch = pos if is_person(src_name) else pos + len("Every ")
            tgt_ch = pos + len(sent) - 1 - len(tgt_name)
            period_ch = pos + len(sent) - 1
            first = tok_at(pos)  # the sentence's first token ("Every" or the person)
            last = tok_at(period_ch)
            src_first, tgt_first = tok_at(src_ch), tok_at(tgt_ch)
            rest = [i for i in range(first, last + 1) if i not in (src_first, tgt_first, last) and i != 0]
            src_anchor = None if src_first == 0 else src_first
            slots.append((src_anchor, tgt_first, last, *rest))
            pos += len(sent)
        # the question: " Is Root a c1 or c2?"
        pos += 1
        root_name = self.names[self.root]
        c1, c2 = self.names[self.cands[0]], self.names[self.cands[1]]
        root_ch = pos + len("Is ")
        c1_ch = root_ch + len(root_name) + len(" a ")
        c2_ch = c1_ch + len(c1) + len(" or ")
        nq = len(q_ids)
        start = nq
        latents = list(range(nq + 1, nq + 1 + self.K))
        end = nq + 1 + self.K
        n = nq + len(lat) + len(pre)
        L = {
            "slots": slots, "sink": 0,
            "root": tok_at(root_ch), "c1": tok_at(c1_ch), "c2": tok_at(c2_ch), "q": tok_at(pos),
            "start": start, "latents": latents, "end": end, "a": n - 1, "n": n,
            "thought1_query": start,
            "prefix_len": len(pre),
        }
        extra = {f"latent{k}": [latents[k]] for k in range(self.K)}
        extra["search_latent"] = latents[: self.L - 1]
        L["extra_query_classes"] = extra
        return L

    # --- graph helpers with L instead of K ------------------------------------
    def parent_slots(self):
        d = self.depths()
        return [j for j, (s, t) in enumerate(self.edges) if t == self.target and d.get(s) == self.L - 1]


# --- sets, controls and donors (the common.py versions compare node ids and K) --

def recipient_prompts_nl(mode):
    return [(gi, s, NLPrompt.from_sample(s)) for gi, s in recipients(mode)]


def permuted(pr, rng):
    E = len(pr.edges)
    perm = list(range(E))
    for _ in range(100):
        rng.shuffle(perm)
        if any(perm[j] != j for j in range(E)):
            break
    return pr.with_edges([pr.edges[j] for j in perm])


def reserialized_nl(pr, gi, base_seed):
    """The same graph in a second, pinned sentence order."""
    return permuted(pr, random.Random(test_pin(gi, base_seed, True)))


def random_donor_nl(train, L, rng, exclude=()):
    for _ in range(100000):
        gi = rng.randrange(len(train))
        if gi in exclude or len(train[gi]["steps"]) != L:
            continue
        return gi, train[gi]
    raise RuntimeError(f"no training graph with L={L}")


def same_answer_donor_nl(train, pr, exclude=()):
    """Training graph with the same target name, decoy name and solution
    length. (gi, sample) or None."""
    t, d = pr.names[pr.target], pr.names[pr.decoy]
    for j, s in enumerate(train):
        if j in exclude or len(s["steps"]) != pr.L:
            continue
        sym = s["idx_to_symbol"]
        if sym[s["target"]] == t and sym[s["neg_target"]] == d:
            return j, s
    return None


def donor_run_nl(runner, train, gi, attn_eager=False):
    from measure import capture
    pr = NLPrompt.from_sample(train[gi])
    return pr, capture(runner, pr.ids(runner.tok), attn_eager=attn_eager)
