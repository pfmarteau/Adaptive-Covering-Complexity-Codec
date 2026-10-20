#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared corpus generators for ablation.py, pareto.py and bench_codec.py   (corpora_2 = corpora_1 + extensions).

Compatibility: with default settings, make()/genomic()/json_logs() return EXACTLY the same
corpora as corpora_1.py (verified seed by seed). Additions are optional:

  * CORPORA_GENOMIC_REPS="lo,hi"  (environment variable, read at import): range for the motif repetition count.
    Default 4,100 (like corpora_1). Note: 4,100 yields ~292 bytes/record, i.e., ~292 KB for 1000 records,
    whereas the paper announces ~142 KB; a range like 4,36 yields ~142 bytes/record.
  * CORPORA_N_MOTIFS="N": size of the motif pool (default 100). The template and the first k motifs are
    identical regardless of N (same private RNG stream).
  * source_entropy_bits(kind): exact Shannon entropy of the source, in bits per record. This is an asymptotic
    floor for any decodable code (excluding dictionary/model cost), useful for situating a codec.

Train and test sets are always drawn with independent seeds (seed and seed + TEST_SEED_OFFSET), never from the
same RNG stream, and never shared.
"""

import math
import os
import random
from collections import Counter
from typing import Callable, Dict, List, Tuple

TEST_SEED_OFFSET = 10_000

GENOMIC_REPS = tuple(int(x) for x in os.environ.get("CORPORA_GENOMIC_REPS", "4,100").split(","))
N_MOTIFS = int(os.environ.get("CORPORA_N_MOTIFS", "100"))
assert len(GENOMIC_REPS) == 2 and 1 <= GENOMIC_REPS[0] <= GENOMIC_REPS[1]

# Constants are derived from a fixed private RNG so they are identical in every
# process (needed because pareto.py re-imports this module in spawned children).
_CONST = random.Random(20260101)
TEMPLATE = "".join(_CONST.choice("ACGT") for _ in range(52))

MOTIFS = [
    "".join(_CONST.choice("ACGT") for _ in range(_CONST.choice((3, 4, 5, 6))))
    for _ in range(N_MOTIFS)
]

SERVERS = [f"node-{k}" for k in range(5)]
STATUS = [200, 200, 200, 200, 201, 404, 500]
EVENTS = ["USER_LOGIN_OK", "USER_LOGOUT", "TOKEN_REFRESH",
          "PASSWORD_RESET", "SESSION_EXPIRED"]


def genomic(n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    lo, hi = GENOMIC_REPS
    out = []
    for _ in range(n):
        motif = rng.choice(MOTIFS)
        out.append(TEMPLATE + motif * rng.randint(lo, hi))
    return out


def json_logs(n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        m, s = rng.randrange(60), rng.randrange(60)
        out.append(
            '{"timestamp":"2026-10-04T10:%02d:%02dZ", "server":"%s", '
            '"status":%d, "payload":"%s_SESSION_ID_%d"}'
            % (m, s, rng.choice(SERVERS), rng.choice(STATUS),
               rng.choice(EVENTS), rng.randrange(10 ** 6))
        )
    return out


GENERATORS: Dict[str, Callable[[int, int], List[str]]] = {
    "genomic": genomic,
    "json": json_logs,
}


def _H(seq) -> float:
    """Entropy (bits) of the uniform draw of an element from `seq` (duplicates weigh more)."""
    c, n = Counter(seq), len(seq)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def source_entropy_bits(kind: str) -> float:
    """Exact Shannon entropy (bits/record) of the source, assuming injective generation
    (true for default settings: no pair (motif, k) yields the same string)."""
    if kind == "genomic":
        lo, hi = GENOMIC_REPS
        return _H(MOTIFS) + math.log2(hi - lo + 1)
    if kind == "json":
        return (2 * math.log2(60) + _H(SERVERS) + _H(STATUS) + _H(EVENTS) + math.log2(10 ** 6))
    raise KeyError(kind)


def make(kind: str, n_train: int, n_test: int, seed: int) -> Tuple[List[str], List[str]]:
    """Return (train, test) drawn with independent seeds."""
    gen = GENERATORS[kind]
    return gen(n_train, seed), gen(n_test, seed + TEST_SEED_OFFSET)