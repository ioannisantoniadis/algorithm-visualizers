"""
algorithm.py — from-scratch scaled dot-product attention

This implements exactly the mechanism from "Attention Is All You Need":

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V

with Q = X Wq, K = X Wk, V = X Wv, and a final output projection Wo that
projects each head's result back into the embedding space before summing
across heads (the residual "multi-head attention" block).

The twist — and the honest thing to say about it — is that Wq, Wk, Wv, Wo
are NOT learned. There is no training loop anywhere in this repository.
Each head's weight matrices are hand-constructed *selectors* that isolate
one feature block of the embedding built in data.py:

  Entity-linking head — reads only the entity-id block. Q and K are
      identical one-hot vectors, so QKᵀ is 1 exactly where two tokens share
      an entity id and 0 otherwise. This is what makes "it" attend to its
      antecedent and an adjective attend to the noun it modifies.

  Syntax head — reads only the role block, but K is passed through a
      hand-picked "role affinity" matrix A before the dot product, so
      scores[i, j] = A[role_i, role_j]. A encodes ordinary dependency
      intuitions (verbs and determiners attend to nouns, adjectives attend
      to nouns, etc.) chosen by inspection, not fit to any corpus.

  Position head — reads only the sinusoidal position block with an
      unmodified dot product, giving a locality bias "for free" from the
      geometry of sinusoids (nearby positions are more aligned).

  Random head (optional 4th head) — genuinely random, untrained Wq/Wk/Wv/Wo
      over the *full* embedding. It exists as a control: this is what
      attention looks like when nothing has been designed or trained into
      it, for contrast with the three heads above.

Every projection, score matrix, softmax and weighted sum is recorded as a
Snapshot so the visualiser can play back the full forward pass step by
step, exactly like the E/M-step snapshots in kmeans-viz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .data import ENTITY_DIM, ROLE_DIM, ROLES

HeadKind = Literal["entity", "role", "position", "random"]

# ---------------------------------------------------------------------------
# Hand-picked syntactic role affinity matrix
#
# Rows = query role, columns = key role. Bigger value => stronger attention
# from that query role to that key role. Chosen by hand to reflect ordinary
# dependency-grammar intuition (determiners/adjectives/verbs look at nouns;
# nouns look at their verb); NOT fit to any text.
# ---------------------------------------------------------------------------

_R = {r: i for i, r in enumerate(ROLES)}
ROLE_AFFINITY = np.zeros((ROLE_DIM, ROLE_DIM))
_affinities = {
    ("DET", "NOUN"): 4, ("DET", "PRON"): 3, ("DET", "DET"): 0,
    ("NOUN", "VERB"): 3, ("NOUN", "ADJ"): 2, ("NOUN", "NOUN"): 1, ("NOUN", "DET"): 1,
    ("VERB", "NOUN"): 4, ("VERB", "PRON"): 3, ("VERB", "OTHER"): 1, ("VERB", "VERB"): 1,
    ("ADJ", "NOUN"): 4, ("ADJ", "PRON"): 3, ("ADJ", "ADJ"): 1,
    ("PRON", "VERB"): 3, ("PRON", "NOUN"): 1, ("PRON", "PRON"): 1,
    ("OTHER", "VERB"): 2, ("OTHER", "NOUN"): 1, ("OTHER", "OTHER"): 1,
}
for (q, k), v in _affinities.items():
    ROLE_AFFINITY[_R[q], _R[k]] = v


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HeadSpec:
    """Fixed, hand-built (or random) weight matrices for one attention head.

    `temperature` scales the scores right before softmax (equivalently, it
    scales how "confident" this hand-built rule is allowed to be). A
    learned model's weight magnitudes implicitly do the same job; since
    nothing here is learned, we pick a fixed temperature by hand so each
    head's pattern is legible rather than washed out by the softmax of a
    small, mostly-zero score matrix. This is the one deliberately "tuned"
    constant in the whole project — everything else falls out of the
    feature design.
    """

    name: str
    kind: HeadKind
    description: str
    Wq: np.ndarray  # (D, d_k)
    Wk: np.ndarray  # (D, d_k)
    Wv: np.ndarray  # (D, d_k)
    Wo: np.ndarray  # (d_k, D) — projects the head's output back to embedding space
    d_k: int
    temperature: float = 1.0


@dataclass
class HeadResult:
    """Every intermediate matrix produced by running one head, kept around
    so the visualiser can show any stage without recomputation."""

    spec: HeadSpec
    Q: np.ndarray
    K: np.ndarray
    V: np.ndarray
    scores_raw: np.ndarray      # Q Kᵀ
    scores_scaled: np.ndarray  # Q Kᵀ / √d_k
    weights: np.ndarray        # softmax(scores_scaled), rows sum to 1
    head_output: np.ndarray    # weights @ V, in the head's own d_k-dim space
    output_full: np.ndarray    # head_output @ Wo, back in embedding space

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass
class Snapshot:
    """One frame of the step-through / auto-play walkthrough.

    `phase` selects what stage of the pipeline this frame illustrates.
    `head_idx` / `substep` are only meaningful when phase == "head".
    All snapshots share references to the same `results` list — nothing is
    recomputed or duplicated per frame, only the "which stage to look at"
    pointer changes.
    """

    phase: Literal["embed", "head", "combine"]
    title: str
    description: str
    tokens: list[str]
    embeddings: np.ndarray
    results: list[HeadResult]
    head_idx: int = -1
    substep: str = ""  # "qkv" | "scores" | "scale" | "softmax" | "output"
    context: np.ndarray | None = None


SUBSTEPS = ["qkv", "scores", "scale", "softmax", "output"]


# ---------------------------------------------------------------------------
# Head construction
# ---------------------------------------------------------------------------

def _selector(total_dim: int, start: int, size: int) -> np.ndarray:
    """(total_dim, size) matrix that extracts columns [start, start+size)."""
    P = np.zeros((total_dim, size))
    P[start:start + size, :] = np.eye(size)
    return P


def build_head_specs(
    blocks,
    n_heads: int,
    seed: int = 0,
    random_head_dim: int = 4,
) -> list[HeadSpec]:
    """Build up to 4 heads, in a fixed order, using the block layout from
    data.make_embeddings. `n_heads` (1-4) truncates the list — this is what
    the sidebar's "number of heads" slider controls.
    """
    D = blocks.total_dim
    r0, r1 = blocks.role
    e0, e1 = blocks.entity
    p0, p1 = blocks.position

    P_role = _selector(D, r0, r1 - r0)
    P_entity = _selector(D, e0, e1 - e0)
    P_pos = _selector(D, p0, p1 - p0)

    heads: list[HeadSpec] = [
        HeadSpec(
            name="Entity-linking",
            kind="entity",
            description=(
                "Reads only the entity-id block. Query and Key are the same "
                "one-hot vector, so a token attends exactly to whatever "
                "shares its entity id — this is how a pronoun finds its "
                "antecedent noun, and how an adjective finds the noun it "
                "modifies."
            ),
            Wq=P_entity, Wk=P_entity, Wv=P_entity, Wo=P_entity.T,
            d_k=ENTITY_DIM, temperature=6.0,
        ),
        HeadSpec(
            name="Syntax",
            kind="role",
            description=(
                "Reads only the syntactic-role block. Keys are passed "
                "through a hand-picked role-affinity matrix before the dot "
                "product, so attention follows ordinary dependency "
                "intuitions (verbs/determiners/adjectives look at nouns)."
            ),
            Wq=P_role, Wk=P_role @ ROLE_AFFINITY.T, Wv=P_role, Wo=P_role.T,
            d_k=ROLE_DIM, temperature=3.0,
        ),
        HeadSpec(
            name="Position",
            kind="position",
            description=(
                "Reads only the sinusoidal position block with a plain dot "
                "product. Sinusoids of nearby positions point in similar "
                "directions, so this head ends up favouring nearby tokens "
                "purely from geometry — no notion of syntax or meaning."
            ),
            Wq=P_pos, Wk=P_pos, Wv=P_pos, Wo=P_pos.T,
            d_k=p1 - p0, temperature=4.0,
        ),
    ]

    if n_heads >= 4:
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(D)
        Wq = rng.normal(0, scale, size=(D, random_head_dim))
        Wk = rng.normal(0, scale, size=(D, random_head_dim))
        Wv = rng.normal(0, scale, size=(D, random_head_dim))
        Wo = rng.normal(0, scale, size=(random_head_dim, D))
        heads.append(
            HeadSpec(
                name="Random (untrained)",
                kind="random",
                description=(
                    "A genuinely random, untrained head over the full "
                    "embedding — included as a control. This is what "
                    "attention looks like with no design or training at "
                    "all: no consistent story, just whatever the random "
                    "weights happen to align with."
                ),
                Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, d_k=random_head_dim,
            )
        )

    return heads[: max(1, min(4, n_heads))]


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def run_attention(X: np.ndarray, head_specs: list[HeadSpec]) -> list[HeadResult]:
    """Run every head's scaled dot-product attention over X and keep every
    intermediate matrix."""
    results = []
    for spec in head_specs:
        Q = X @ spec.Wq
        K = X @ spec.Wk
        V = X @ spec.Wv
        scores_raw = Q @ K.T
        scores_scaled = (scores_raw / np.sqrt(spec.d_k)) * spec.temperature
        weights = _softmax(scores_scaled, axis=-1)
        head_output = weights @ V
        output_full = head_output @ spec.Wo
        results.append(
            HeadResult(
                spec=spec, Q=Q, K=K, V=V,
                scores_raw=scores_raw, scores_scaled=scores_scaled,
                weights=weights, head_output=head_output, output_full=output_full,
            )
        )
    return results


def combine_heads(X: np.ndarray, results: list[HeadResult]) -> np.ndarray:
    """Residual combination: original embedding + sum of every head's
    contribution, each written back into the embedding subspace it read
    from. (Real Transformers also apply LayerNorm here; omitted for
    clarity — see README limitations.)
    """
    context = X.copy()
    for r in results:
        context = context + r.output_full
    return context


# ---------------------------------------------------------------------------
# Snapshot construction — the step-through / auto-play script
# ---------------------------------------------------------------------------

_SUBSTEP_TITLES = {
    "qkv": "Project to Q, K, V",
    "scores": "Raw scores  QKᵀ",
    "scale": "Scale by 1/√d_k",
    "softmax": "Softmax → attention weights",
    "output": "Weighted sum of V",
}

_SUBSTEP_DESCRIPTIONS = {
    "qkv": (
        "Q = X·Wq, K = X·Wk, V = X·Wv. {desc}"
    ),
    "scores": (
        "scores[i, j] = Qᵢ · Kⱼ — a raw similarity between every query "
        "token i and every key token j. Larger = more aligned."
    ),
    "scale": (
        "Dividing by √d_k = √{dk} keeps the scores in a range where softmax "
        "doesn't saturate as the head dimension grows — 'scaled' dot-product "
        "attention. A fixed temperature is then applied so this hand-built "
        "head's pattern is legible (a learned model's weight magnitudes do "
        "the equivalent job automatically)."
    ),
    "softmax": (
        "Each row is turned into a probability distribution over key "
        "tokens: row i is how much token i attends to every token, "
        "summing to 1."
    ),
    "output": (
        "Each token's new representation (in this head) is the attention-"
        "weighted average of the Value vectors — a blend of whichever "
        "tokens it attended to."
    ),
}


def make_snapshots(
    tokens: list[str],
    X: np.ndarray,
    results: list[HeadResult],
) -> list[Snapshot]:
    """Build the ordered list of Snapshots: embed -> (per head: qkv, scores,
    scale, softmax, output) -> combine.
    """
    snapshots: list[Snapshot] = [
        Snapshot(
            phase="embed",
            title="Token embeddings",
            description=(
                "Each token's embedding concatenates three hand-designed, "
                "untrained feature blocks: syntactic role (one-hot), entity "
                "id (one-hot, links referring expressions), and position "
                "(sinusoidal). Nothing here was learned from data."
            ),
            tokens=tokens,
            embeddings=X,
            results=results,
        )
    ]

    n_heads = len(results)
    for h, r in enumerate(results):
        for substep in SUBSTEPS:
            desc = _SUBSTEP_DESCRIPTIONS[substep].format(desc=r.spec.description, dk=r.spec.d_k)
            snapshots.append(
                Snapshot(
                    phase="head",
                    title=f"Head {h + 1}/{n_heads} · {r.name} — {_SUBSTEP_TITLES[substep]}",
                    description=desc,
                    tokens=tokens,
                    embeddings=X,
                    results=results,
                    head_idx=h,
                    substep=substep,
                )
            )

    context = combine_heads(X, results)
    snapshots.append(
        Snapshot(
            phase="combine",
            title="Combine heads → final context vectors",
            description=(
                "Every head's output is projected back into the embedding "
                "space (via its own Wo) and added to the original "
                "embedding — a residual connection. The result is a new, "
                "context-aware vector for every token."
            ),
            tokens=tokens,
            embeddings=X,
            results=results,
            context=context,
        )
    )
    return snapshots
