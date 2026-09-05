"""Thought-edit constructors that take explicit directions or bases.

Runner (harness.py) holds the node-indexed single-direction operations. The
constructors here take unit vectors or orthonormal bases directly, so they
work with any basis: input embeddings, probe directions, INLP subspaces, or
J-lens directions. Every edit rescales its output to the pre-edit thought norm.

Two calling conventions are kept as they were in the original callers:
  f(t)        -- span_project_out, sub_dir, sub_dir_random_matched, swap_edit
  f(t, info)  -- edit_subtract_subspace (matches Runner.measure's edit arg)
"""

import torch


# --- span removal ------------------------------------------------------------

def span_project_out(Q):
    """Edit: remove the span of orthonormal columns Q from t, rescale to ||t||."""
    def f(t):
        t2 = t - Q @ (Q.T @ t)
        nrm = t2.norm()
        if nrm < 1e-8:  # t entirely inside the span; cannot renormalize
            return t
        return t2 / nrm * t.norm()
    return f


def orthonormal(runner, vecs):
    """Orthonormal basis (d, m) for the span of a list of unit vectors."""
    A = torch.stack(vecs, dim=1)  # (d, m)
    Q, _ = torch.linalg.qr(A)
    return Q


def random_orthonormal(runner, m, d):
    """m random orthonormal directions in R^d, drawn from runner.gen."""
    A = torch.randn(d, m, generator=runner.gen)
    Q, _ = torch.linalg.qr(A)
    return Q.to(runner.device)


def rand_orthonormal(k, dim, gen, dev):
    """k random orthonormal directions in R^dim from an explicit generator
    (rank-matched random control for a k-dimensional removal)."""
    R = torch.randn(dim, k, generator=gen)
    Q, _ = torch.linalg.qr(R)
    return Q[:, :k].contiguous().to(dev)


def edit_subtract_subspace(B):
    """Norm-preserving projection out of span(B) (B: (dim,k) orthonormal, on
    the model device). Signature f(t, info) for Runner.measure."""
    def f(t, info):
        t2 = t - B @ (B.t() @ t)
        return t2 / t2.norm() * t.norm()
    return f


def per_pass_dirs(runner, sample, skips):
    """{pass_idx: [u_hat(v) for v in neighbor_k[pass_idx+1]]}, skipping nodes
    without a direction in the runner's basis (counted in skips)."""
    n_pass = len(sample["steps"])
    dirs = {}
    for k in range(n_pass):
        nodes = sample["neighbor_k"].get(str(k + 1), [])
        if not nodes:
            skips["missing_depth_key"] += 1
            continue
        us = []
        for v in nodes:
            try:
                us.append(runner.u_hat(v))
            except ValueError:
                skips["missing_direction"] += 1
        if us:
            dirs[k] = us
        else:
            skips["pass_all_directions_missing"] += 1
    return dirs


# --- single explicit directions ----------------------------------------------

def sub_dir(u):
    """Norm-preserving subtract of unit direction u."""
    def f(t):
        t2 = t - (t @ u) * u
        return t2 / t2.norm().clamp_min(1e-12) * t.norm()
    return f


def sub_dir_random_matched(u, r):
    """Same coefficient magnitude as sub_dir(u), removed along random unit r."""
    def f(t):
        t2 = t - (t @ u) * r
        return t2 / t2.norm().clamp_min(1e-12) * t.norm()
    return f


def swap_edit(u_a, u_b):
    """Concept swap: V=[u_a u_b], c=V^+ t, swap the two coordinates, add back;
    then norm-preserving rescale. Returns None if directions are collinear."""
    if abs(float(u_a @ u_b)) > 0.999:
        return None
    V = torch.stack([u_a, u_b], dim=1)  # (768, 2)
    Vp = torch.linalg.pinv(V)  # (2, 768)

    def f(t):
        c = Vp @ t
        t2 = t + V @ (c.flip(0) - c)
        return t2 / t2.norm().clamp_min(1e-12) * t.norm()
    return f


# --- J-lens basis loader -----------------------------------------------------

class JBasis:
    """Loads jlens_basis.pt from fit_jlens.py: tensor (K, vocab, 768), zero
    rows meaning 'no direction for this node at this pass'."""

    def __init__(self, path):
        self.J = torch.load(path, map_location="cpu", weights_only=True)

    def to(self, device):
        self.J = self.J.to(device)
        return self

    def dir(self, v, k):
        if k >= self.J.shape[0]:
            raise ValueError(f"no pass {k} in J basis")
        d = self.J[k, v]
        if d.norm() == 0:
            raise ValueError(f"no J direction for node {v} at pass {k}")
        return d / d.norm()
