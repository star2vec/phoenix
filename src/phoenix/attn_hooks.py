"""Attention and residual-stream hooks for the two-layer model.

Off by default: nothing here is installed unless a driver enters one of the
two context managers, so tests/test_fast_equivalence.py stays bit-exact.

AttnHooks wraps the eager `_attn` method of each GPT-2 attention layer to
  (i)   record attention weights per layer and query position,
  (ii)  overwrite cached keys or values at chosen (layer, head, position),
  (iii) add an attention mask for chosen query -> key pairs.
The SDPA attention class falls back to the eager path whenever attention
weights are requested, so FastCoconut must run with attn_eager=True for these
hooks to see any traffic (drivers pass it; the equivalence test reports the
eager-vs-SDPA rounding difference).

ResidualHooks records or overwrites the residual stream by level and
absolute position. Level 0 is the input to layer 1 (token or recycled
thought embedding plus position embedding); level l is the output of block l
(l = 1..n_layer). It uses module hooks and works on either attention path.

Position convention (both classes): the cache always starts at position 0,
so a key index is an absolute prompt position, and query row i of a call at
cache length L with q rows is absolute position (L - q) + i. Every position
is computed exactly once across the passes of one forward, so recorded
entries are unique per (layer, position).
"""

import torch

MASK_VALUE = -1e9  # large negative, added before the softmax (never -inf)


class AttnHooks:
    def __init__(self, base):
        """base: the GPT2LMHeadModel (runner.model.base_causallm)."""
        self.blocks = list(base.transformer.h)
        self.n_layers = len(self.blocks)
        self.kv_patches = []
        self.masks = []
        self.record_weights = False
        self.record_kv = False
        self.weights = []   # dicts: layer, offset, w (heads, q_len, k_len)
        self.kv = {}        # layer -> (key, value), each (heads, k_len, head_dim)

    # --- configuration ---------------------------------------------------
    def add_kv_patch(self, layer, which, positions, tensor, heads=None):
        """Overwrite keys ('k') or values ('v') of `layer` at absolute
        `positions` with `tensor` of shape (n_heads_selected, len(positions),
        head_dim). heads=None means all heads, in order."""
        assert which in ("k", "v")
        n_heads = self.blocks[layer].attn.num_heads if heads is None else len(heads)
        assert tuple(tensor.shape[:2]) == (n_heads, len(positions)), tensor.shape
        self.kv_patches.append({
            "layer": layer, "which": which, "positions": list(positions),
            "tensor": tensor, "heads": None if heads is None else list(heads),
        })

    def add_mask(self, layers, q_positions, k_positions, heads=None):
        """Block attention from each query position to each key position in
        the given layers, for the given heads (None = all heads)."""
        self.masks.append({"layers": set(layers), "q": list(q_positions), "k": list(k_positions),
                           "heads": None if heads is None else list(heads)})

    def clear_records(self):
        self.weights = []
        self.kv = {}

    def clear(self):
        self.kv_patches = []
        self.masks = []
        self.clear_records()

    def active(self):
        return bool(self.kv_patches or self.masks or self.record_weights or self.record_kv)

    # --- install / remove ------------------------------------------------
    def __enter__(self):
        for li, blk in enumerate(self.blocks):
            orig = blk.attn.__class__._attn.__get__(blk.attn)  # class method, bound
            blk.attn._attn = self._wrap(li, orig)
        return self

    def __exit__(self, *exc):
        for blk in self.blocks:
            blk.attn.__dict__.pop("_attn", None)  # restore class lookup
        return False

    def _wrap(self, li, orig):
        hooks = self

        def _attn(query, key, value, attention_mask=None, head_mask=None):
            if not hooks.active():
                return orig(query, key, value, attention_mask, head_mask)
            assert query.shape[0] == 1, "AttnHooks assume batch size 1"
            k_len, q_len = key.shape[-2], query.shape[-2]
            offset = k_len - q_len

            cloned = False
            for p in hooks.kv_patches:
                if p["layer"] != li:
                    continue
                sel = [(i, pos) for i, pos in enumerate(p["positions"]) if pos < k_len]
                if not sel:
                    continue
                if not cloned:
                    key, value = key.clone(), value.clone()
                    cloned = True
                tgt = key if p["which"] == "k" else value
                idx = [i for i, _ in sel]
                pos = [pos for _, pos in sel]
                src = p["tensor"].to(device=tgt.device, dtype=tgt.dtype)
                heads = p["heads"] if p["heads"] is not None else list(range(tgt.shape[1]))
                for hi, h in enumerate(heads):
                    tgt[0, h, pos, :] = src[hi, idx, :]

            add = None
            n_h = query.shape[1]
            for m in hooks.masks:
                if li not in m["layers"]:
                    continue
                ks = [k for k in m["k"] if k < k_len]
                if not ks:
                    continue
                hs = list(range(n_h)) if m.get("heads") is None else m["heads"]
                for q in m["q"]:
                    qi = q - offset
                    if 0 <= qi < q_len:
                        if add is None:
                            add = torch.zeros(1, n_h, q_len, k_len, dtype=query.dtype, device=query.device)
                        for hh in hs:
                            add[0, hh, qi, ks] = MASK_VALUE
            if add is not None:
                attention_mask = add if attention_mask is None else attention_mask + add

            out, w = orig(query, key, value, attention_mask, head_mask)

            if hooks.record_weights:
                hooks.weights.append({"layer": li, "offset": offset, "w": w[0].detach().clone()})
            if hooks.record_kv:
                cur = hooks.kv.get(li)
                if cur is None or cur[0].shape[-2] < k_len:
                    hooks.kv[li] = (key[0].detach().clone(), value[0].detach().clone())
            return out, w

        return _attn

    # --- readers ---------------------------------------------------------
    def attention(self, layer, q_abs):
        """(heads, k_len) attention weights of the query at absolute position
        q_abs in `layer`, from the recorded pass that computed it."""
        for rec in self.weights:
            if rec["layer"] != layer:
                continue
            q_len = rec["w"].shape[1]
            if rec["offset"] <= q_abs < rec["offset"] + q_len:
                return rec["w"][:, q_abs - rec["offset"], :]
        raise KeyError(f"no recorded attention for layer {layer}, position {q_abs}")

    def kv_at(self, layer, which, positions, heads=None):
        """(n_heads_selected, len(positions), head_dim) slice of the recorded
        keys ('k') or values ('v'), ready for add_kv_patch."""
        t = self.kv[layer][0 if which == "k" else 1]
        if heads is not None:
            t = t[list(heads)]
        return t[:, list(positions), :].clone()


class ResidualHooks:
    def __init__(self, base):
        self.blocks = list(base.transformer.h)
        self.patches = {}   # (level, abs_pos) -> (d_model,) tensor
        self.record = False
        self.store = {}     # level -> {abs_pos: (d_model,) tensor}
        self._handles = []

    @staticmethod
    def _offset(kwargs):
        lp = kwargs.get("layer_past")
        return 0 if lp is None else int(lp[0].shape[-2])

    def _apply(self, level, hidden, offset):
        if not self.record and not self.patches:
            return hidden  # inactive: touch nothing (keeps the bit-exact path)
        assert hidden.shape[0] == 1, "ResidualHooks assume batch size 1"
        q_len = hidden.shape[1]
        if self.record:
            st = self.store.setdefault(level, {})
            for i in range(q_len):
                st[offset + i] = hidden[0, i].detach().clone()
        todo = [(pos, v) for (lvl, pos), v in self.patches.items()
                if lvl == level and offset <= pos < offset + q_len]
        if todo:
            hidden = hidden.clone()
            for pos, v in todo:
                hidden[0, pos - offset] = v.to(device=hidden.device, dtype=hidden.dtype)
        return hidden

    def __enter__(self):
        def pre(mod, args, kwargs):
            h = self._apply(0, args[0], self._offset(kwargs))
            return (h,) + tuple(args[1:]), kwargs

        self._handles.append(self.blocks[0].register_forward_pre_hook(pre, with_kwargs=True))
        for li, blk in enumerate(self.blocks):
            def post(mod, args, kwargs, output, li=li):
                h = self._apply(li + 1, output[0], self._offset(kwargs))
                return (h,) + tuple(output[1:])

            self._handles.append(blk.register_forward_hook(post, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def levels(self):
        return list(range(len(self.blocks) + 1))


class MLPHooks:
    """Record or replace the MLP output of each block at chosen absolute
    positions. patches[(layer, abs_pos)] = (d_model,) tensor replaces the MLP
    output (before it is added to the residual). Inactive when nothing is
    set, so the bit-exact path is untouched."""

    def __init__(self, base):
        self.blocks = list(base.transformer.h)
        self.patches = {}
        self.record = False
        self.store = {}     # layer -> {abs_pos: (d_model,)}
        self._handles = []
        self._offset = 0

    def set_offset(self, offset):
        """Absolute position of the first row of the current call. The
        block's forward receives layer_past; the pre-hook below reads it."""
        self._offset = offset

    def __enter__(self):
        for li, blk in enumerate(self.blocks):
            def pre(mod, args, kwargs, li=li):
                lp = kwargs.get("layer_past")
                self._offset = 0 if lp is None else int(lp[0].shape[-2])
                return None

            def post(mod, args, output, li=li):
                if not self.record and not self.patches:
                    return None
                h = output
                assert h.shape[0] == 1, "MLPHooks assume batch size 1"
                q_len = h.shape[1]
                if self.record:
                    st = self.store.setdefault(li, {})
                    for i in range(q_len):
                        st[self._offset + i] = h[0, i].detach().clone()
                todo = [(pos, v) for (lvl, pos), v in self.patches.items()
                        if lvl == li and self._offset <= pos < self._offset + q_len]
                if not todo:
                    return None
                h = h.clone()
                for pos, v in todo:
                    h[0, pos - self._offset] = v.to(device=h.device, dtype=h.dtype)
                return h

            self._handles.append(blk.register_forward_pre_hook(pre, with_kwargs=True))
            self._handles.append(blk.mlp.register_forward_hook(post))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False
