"""
data.py — token/sentence generation and hand-designed embeddings

There is no tokenizer or trained embedding table here. Instead, every
token gets a feature-based embedding built from three independent,
hand-designed blocks concatenated together:

  role block      (6 dims, one-hot)  — the token's syntactic role
                                        (DET, NOUN, VERB, ADJ, PRON, OTHER)
  entity block     (3 dims, one-hot) — which "thing" the token refers to
                                        (shared by a noun and the pronoun /
                                        adjective that points back at it;
                                        all-zero if the token has no referent)
  position block   (pos_dim, sinusoidal) — the standard Transformer
                                        sin/cos position encoding

This is the honesty-critical part of the whole project: nothing here is
learned from data. The blocks are a deliberate, transparent stand-in for
what a real trained embedding table + tokenizer would produce, chosen so
that attention.py's hand-built attention heads have something structured
to latch onto. See README.md for the full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Feature vocabulary
# ---------------------------------------------------------------------------

ROLES = ["DET", "NOUN", "VERB", "ADJ", "PRON", "OTHER"]
ROLE_INDEX = {r: i for i, r in enumerate(ROLES)}
ROLE_DIM = len(ROLES)

# Max number of distinct "entities" (things a noun/pronoun/adjective can
# refer to) a sentence can track. Entity id 0 means "refers to nothing in
# particular" and is encoded as the all-zero vector rather than a one-hot,
# so function words don't fight for a slot in this block.
MAX_ENTITIES = 3
ENTITY_DIM = MAX_ENTITIES

# ---------------------------------------------------------------------------
# Sentence templates
#
# Each slot is (options, role, entity_id). `options` is either a fixed
# string (function words) or a list of interchangeable words — the seed
# picks one, so the *surface form* changes between runs while the
# role/entity structure (and therefore the attention story) stays fixed.
# This is what "Re-generate" demonstrates: the mechanism generalises across
# vocabulary as long as the hand-assigned roles/ids stay consistent.
# ---------------------------------------------------------------------------

TEMPLATES: list[dict] = [
    {
        "name": "Coreference — pronoun refers to the subject",
        "slots": [
            ("the", "DET", None),
            (["cat", "dog", "fox"], "NOUN", 1),
            (["sat", "rested", "waited"], "VERB", None),
            ("on", "OTHER", None),
            ("the", "DET", None),
            (["mat", "rug", "step"], "NOUN", 2),
            ("because", "OTHER", None),
            ("it", "PRON", 1),
            ("was", "VERB", None),
            (["tired", "sleepy", "content"], "ADJ", None),
        ],
    },
    {
        "name": "Coreference — pronoun refers to the object",
        "slots": [
            ("the", "DET", None),
            (["cat", "dog", "fox"], "NOUN", 1),
            (["chased", "watched", "followed"], "VERB", None),
            ("the", "DET", None),
            (["ball", "mouse", "leaf"], "NOUN", 2),
            ("because", "OTHER", None),
            ("it", "PRON", 2),
            ("was", "VERB", None),
            (["fast", "bright", "loud"], "ADJ", None),
        ],
    },
    {
        "name": "Adjective-noun modification (no pronoun)",
        "slots": [
            ("the", "DET", None),
            (["girl", "boy", "robot"], "NOUN", 1),
            (["kicked", "threw", "rolled"], "VERB", None),
            ("the", "DET", None),
            (["red", "heavy", "striped"], "ADJ", 2),
            (["ball", "box", "stone"], "NOUN", 2),
        ],
    },
    {
        "name": "Two independent entities (contrast / control)",
        "slots": [
            ("the", "DET", None),
            (["sun", "moon", "storm"], "NOUN", 1),
            (["rose", "faded", "passed"], "VERB", None),
            ("while", "OTHER", None),
            ("the", "DET", None),
            (["river", "valley", "forest"], "NOUN", 2),
            (["slept", "waited", "stayed"], "VERB", None),
        ],
    },
]

TEMPLATE_NAMES = [t["name"] for t in TEMPLATES]


@dataclass
class SentenceData:
    """Hand-tagged sentence: the ground truth a real tokenizer/parser would
    have to *infer* — here it is simply asserted, which is the whole point.
    """

    words: list[str]
    roles: list[str]
    role_ids: list[int]        # index into ROLES, per token
    entity_ids: list[int]      # 0 = no referent, else 1..MAX_ENTITIES


def build_sentence(template_idx: int, seed: int) -> SentenceData:
    """Instantiate a template: pick one option per slot using `seed`.

    The role/entity structure never changes for a given template — only
    the concrete words do. That separation is intentional (see module
    docstring).
    """
    template = TEMPLATES[template_idx % len(TEMPLATES)]
    rng = np.random.default_rng(seed)

    words, roles, entity_ids = [], [], []
    for options, role, entity_id in template["slots"]:
        if isinstance(options, str):
            word = options
        else:
            word = options[int(rng.integers(0, len(options)))]
        words.append(word)
        roles.append(role)
        entity_ids.append(entity_id or 0)

    role_ids = [ROLE_INDEX[r] for r in roles]
    return SentenceData(words=words, roles=roles, role_ids=role_ids, entity_ids=entity_ids)


# ---------------------------------------------------------------------------
# Position encoding
# ---------------------------------------------------------------------------

def sinusoidal_position_encoding(n_positions: int, dim: int) -> np.ndarray:
    """Sin/cos position encoding in the style of Vaswani et al. (2017):

    pe[pos, 2i]   = sin(pos * ω_i)
    pe[pos, 2i+1] = cos(pos * ω_i)

    The original paper spaces frequencies geometrically from 1 down to
    1/10000 — tuned for sequences thousands of tokens long. On a ~10-token
    demo sentence that spacing aliases badly (the fastest frequency
    completes more than half a cycle before the sentence ends, so distant
    positions can spuriously look similar to nearby ones).

    Instead we cap the fastest frequency at π/(n_positions-1): with every
    ω_i at or below that cap, each individual sin/cos pair's dot product
    cos(ω_i · Δ) is monotonically *non-increasing* in the distance
    Δ = |i - j| across the whole sentence, and summing monotonic terms
    keeps the total monotonic. That's what gives the "position head" in
    algorithm.py a clean, honest locality bias instead of an aliased one.
    """
    dim = max(2, dim - (dim % 2))  # force even
    n_freqs = dim // 2
    max_span = max(n_positions - 1, 1)
    omega_max = np.pi / max_span
    omegas = np.linspace(omega_max / n_freqs, omega_max, n_freqs)

    position = np.arange(n_positions)[:, None]
    pe = np.zeros((n_positions, dim))
    pe[:, 0::2] = np.sin(position * omegas)
    pe[:, 1::2] = np.cos(position * omegas)
    return pe


# ---------------------------------------------------------------------------
# Embedding assembly
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingBlocks:
    """Column ranges of each feature block within the embedding matrix."""

    role: tuple[int, int]
    entity: tuple[int, int]
    position: tuple[int, int]

    @property
    def total_dim(self) -> int:
        return self.position[1]


def make_embeddings(sentence: SentenceData, pos_dim: int) -> tuple[np.ndarray, EmbeddingBlocks]:
    """Build the (N, D) embedding matrix for a tagged sentence.

    D = ROLE_DIM + ENTITY_DIM + pos_dim. Every row is the concatenation of:
      - a one-hot role vector
      - a one-hot entity vector (zero vector if entity_id == 0)
      - a sinusoidal position vector
    """
    n = len(sentence.words)
    pos_dim = max(2, pos_dim - (pos_dim % 2))

    role_block = np.zeros((n, ROLE_DIM))
    role_block[np.arange(n), sentence.role_ids] = 1.0

    entity_block = np.zeros((n, ENTITY_DIM))
    for i, eid in enumerate(sentence.entity_ids):
        if eid > 0:
            entity_block[i, eid - 1] = 1.0

    pos_block = sinusoidal_position_encoding(n, pos_dim)

    X = np.concatenate([role_block, entity_block, pos_block], axis=1)

    blocks = EmbeddingBlocks(
        role=(0, ROLE_DIM),
        entity=(ROLE_DIM, ROLE_DIM + ENTITY_DIM),
        position=(ROLE_DIM + ENTITY_DIM, ROLE_DIM + ENTITY_DIM + pos_dim),
    )
    return X, blocks
