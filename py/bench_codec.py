#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_codec.py: Fair benchmark of the "adaptive covering" codec (cs_adapt_codec.py)
against zlib / zstd / brotli / lzma / PPMd / lrzip, with and without dictionary.

Fairness rules applied:
  R1  Same information: bytes <-> str via latin-1 (1 byte = 1 character), no test->train leakage.
  R2  Same accounting: total = payload + index + dictionary/header, charged at its RAW (uncompressed) size for
      EVERY method by default (--dict-charge raw; codec = whole uncompressed header: alphabet + S + tables).
      Stored (zlib) sizes are always reported too (--dict-charge zlib to charge them).
  R3  Same dictionary budget (in bytes) and same training data for all.
  R4  Hyperparameters chosen on a VALIDATION set (never on test), for all.
  R5  Minimal framing for baselines (raw deflate, zstd without magic/checksum/dictID/size).
  R6  Decoding verified across all test data, with decoder reconstructed from serialized bytes.
  R7  Multiple random draws (seeds) + paired bootstrap CI, not a single run.
  R8  Controls: codec ablations (S=empty, without self-reference), cross-dictionary,
      trained order-k context model (isolates the "trained probability tables" effect).

Scenarios: A per-record (main) | B increasing length | C drift | S solid (lrzip, xz...) | T theory
"""
from __future__ import annotations

import argparse
import bz2
import collections
import contextlib
import gc
import importlib.util
import json
import lzma
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import numpy as np
except ImportError:
    np = None
try:
    import zstandard as zstd
except ImportError:
    zstd = None
try:
    import brotli
except ImportError:
    brotli = None
try:
    import pyppmd
except ImportError:
    pyppmd = None

LATIN1 = "latin-1"
INF = float("inf")
DICT_CHARGE = "raw"           # Charging rule applied to EVERY method: "raw" (uncompressed size, default) or "zlib" (stored/compressed size)
REF = "CODEC[S=builder]"      # Reference family; renamed by main() if --mdl is provided
_CACHE: dict = {}             # Already built codecs/MDL (key: seed, modes...) reused across scenarios
PRUNE_NAMES = {"exact": -1, "dead": 0, "lazy": 1, "lazybudget": 2}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def dict_cost(blob: bytes) -> int:
    """Bytes charged for a dictionary/model blob under the active rule (kept for compatibility)."""
    return len(blob) if DICT_CHARGE == "raw" else min(len(blob), len(zlib.compress(blob, 9)))


def dsz(blob: Optional[bytes]) -> dict:
    """Both accountings of a dictionary/model blob: raw size and zlib-compressed size (min(raw, zlib-9)).
    The charged size (Method.dict_bytes) is chosen from them by --dict-charge, for ALL methods alike."""
    if not blob:
        return dict(raw_dict_bytes=0, comp_dict_bytes=0)
    return dict(raw_dict_bytes=len(blob), comp_dict_bytes=min(len(blob), len(zlib.compress(blob, 9))))


def uncompressed_header_size(mod, ctx) -> int:
    """Size of the codec header BEFORE zlib: alphabet + dictionary S + probability/model tables.
    Uses the codec's own get_uncompressed_header_size() (cs_adapt_codec) and falls back to the same parsing."""
    fn = getattr(mod, "get_uncompressed_header_size", None)
    if fn is not None:
        return int(fn(ctx))
    h = ctx.header_bytes
    n = int.from_bytes(h[:4], "big") & 0x7FFFFFFF
    return len(zlib.decompress(h[4:4 + n]))


def varint_len(n: int) -> int:
    k = 1
    while n >= 128:
        n >>= 7
        k += 1
    return k


def vbytes(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------

def load_csv_records(path: str, column: Optional[str], delim: str, header: bool, encoding: str, args=None) -> list[bytes]:
    """By default, EACH ROW of the CSV becomes a JSON record {column: value, ...} (CSV column order).
    With --csv-column X, the record is only the raw value of column X (e.g. column already containing JSON).
    The csv module handles quotes, commas, and line breaks within fields."""
    import csv
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    with open(path, newline="", encoding=encoding, errors="replace") as fh:
        rows = list(csv.reader(fh, delimiter=delim))
    if not rows:
        return []
    names = rows[0] if header else [f"col{i}" for i in range(len(rows[0]))]
    body = rows[1:] if header else rows
    if column is not None:                      # Mode "raw value of a column"
        if column in names:
            idx = names.index(column)
        elif column.lstrip("-").isdigit():
            idx = int(column)
        else:
            raise SystemExit(f"[csv] Column {column!r} not found among {names[:12]}")
        log(f"[csv] Raw column: {names[idx]!r} (index {idx}), {len(body)} rows")
        return [r[idx].encode("utf-8") for r in body if len(r) > idx]
    if len(names) == 1:                         # Single column: raw value
        log(f"[csv] 1 column ({names[0]!r}): raw value, {len(body)} rows")
        return [r[0].encode("utf-8") for r in body if r]
    # --- row -> JSON object ---
    keep = list(range(len(names)))
    if args is not None and args.csv_columns:
        want = [c.strip() for c in args.csv_columns.split(",")]
        missing = [c for c in want if c not in names]
        if missing:
            raise SystemExit(f"[csv] Unknown columns {missing}; available: {names}")
        keep = [names.index(c) for c in want]
    if args is not None and args.csv_drop:
        drop = {c.strip() for c in args.csv_drop.split(",")}
        missing = [c for c in drop if c not in names]
        if missing:
            raise SystemExit(f"[csv] Unknown columns {missing}; available: {names}")
        keep = [i for i in keep if names[i] not in drop]
    keys, seen = [], {}
    for i in keep:                              # Duplicate names -> suffix
        k = names[i]
        seen[k] = seen.get(k, 0) + 1
        keys.append(k if seen[k] == 1 else f"{k}_{seen[k]}")
    infer = bool(args and args.csv_infer_types)
    seps = (",", ":") if (args and args.json_compact) else (", ", ": ")
    int_re = re.compile(r"^-?(0|[1-9][0-9]*)$")

    def val(x: str):
        return int(x) if infer and int_re.match(x) else x

    out = []
    for r in body:
        r = r + [""] * (len(names) - len(r))    # Short rows padded
        out.append(json.dumps({k: val(r[i]) for k, i in zip(keys, keep)}, ensure_ascii=False,
                              separators=seps).encode("utf-8"))
    log(f"[csv] Row -> JSON object, keys = {keys}")
    if out:
        log(f"[csv] Example: {out[0][:200].decode('utf-8', 'replace')}")
    return out


def load_records(path: str, fmt: str, fasta_headers: bool, crop: int, args=None) -> list[bytes]:
    if os.path.isdir(path):
        recs = [open(os.path.join(path, f), "rb").read() for f in sorted(os.listdir(path))]
    elif fmt == "csv":
        recs = load_csv_records(path, args.csv_column, "\t" if args.csv_delimiter == "\\t" else args.csv_delimiter,
                                not args.csv_no_header, args.csv_encoding, args)
    else:
        raw = open(path, "rb").read()
        if fmt == "fasta":
            recs = []
            for chunk in raw.split(b">")[1:]:
                head, _, seq = chunk.partition(b"\n")
                seq = seq.replace(b"\n", b"").replace(b"\r", b"")
                recs.append(head + b"\n" + seq if fasta_headers else seq)
        else:
            recs = raw.splitlines()
    if crop:
        recs = [r[:crop] for r in recs]
    return [r for r in recs if r]  # The codec cannot encode empty strings


def synth(kind: str, n: int, seed: int) -> list[bytes]:
    """Toy generators, ONLY for testing the script (not those from the paper)."""
    rng = random.Random(seed)
    if kind == "dna":
        motifs = ["".join(rng.choice("ACGT") for _ in range(rng.randint(12, 40))) for _ in range(12)]
        out = []
        for _ in range(n):
            parts = []
            for _ in range(rng.randint(3, 8)):
                m = rng.choice(motifs)
                if rng.random() < 0.15:
                    i = rng.randrange(len(m))
                    m = m[:i] + rng.choice("ACGT") + m[i + 1:]
                parts.append(m)
                if rng.random() < 0.5:
                    parts.append("".join(rng.choice("ACGT") for _ in range(rng.randint(2, 12))))
            out.append("".join(parts).encode())
        return out
    svc = ["auth", "db", "api", "cache", "billing"]
    msg = ["user login ok", "connection timeout", "cache miss", "payment accepted", "retry scheduled"]
    return [
        ('{"ts": %d, "level": "%s", "svc": "%s", "msg": "%s", "uid": %d, "ip": "10.0.%d.%d"}' % (
            1700000000 + rng.randint(0, 10 ** 6), rng.choice(["INFO", "WARN", "ERROR"]), rng.choice(svc),
            rng.choice(msg), rng.randint(1, 9999), rng.randint(0, 9), rng.randint(0, 255))).encode()
        for _ in range(n)
    ]


def make_split(recs, ntrain, nval, ntest, seed, group_re=None, drop_leaked=True):
    """Disjoint split. With group_re, a group (e.g. gene id) never crosses two splits."""
    rng = random.Random(seed)
    need = ntrain + nval + ntest
    if len(recs) < need:
        f = len(recs) / need
        ntrain, nval, ntest = int(ntrain * f), int(nval * f), int(ntest * f)
        log(f"[split] Corpus too small, quotas reduced: train={ntrain} val={nval} test={ntest}")
    if group_re:
        rx = re.compile(group_re.encode())
        groups = collections.defaultdict(list)
        for r in recs:
            m = rx.search(r)
            groups[m.group(0) if m else r].append(r)
        keys = list(groups)
        rng.shuffle(keys)
        out = {"test": [], "val": [], "train": []}
        quotas = [("test", ntest), ("val", nval), ("train", ntrain)]
        qi = 0
        for k in keys:
            while qi < 3 and len(out[quotas[qi][0]]) >= quotas[qi][1]:
                qi += 1
            if qi == 3:
                break
            name, q = quotas[qi]
            out[name].extend(groups[k][: q - len(out[name])])
        train, val, test = out["train"], out["val"], out["test"]
    else:
        pool = recs[:]
        rng.shuffle(pool)
        test, val, train = pool[:ntest], pool[ntest:ntest + nval], pool[ntest + nval:ntest + nval + ntrain]
    tr = set(train)
    leak = sum(1 for r in test if r in tr) / max(1, len(test))
    if drop_leaked:
        test = [r for r in test if r not in tr]
        val = [r for r in val if r not in tr]
    return train, val, test, leak


def gen_split(gmod, kind: str, ntrain: int, nval: int, ntest: int, seed: int, drop_leaked: bool = True):
    """Paper protocol with generator module: train = gen(seed), test = gen(seed + TEST_SEED_OFFSET)
    (independent RNG streams). Validation is the CONTINUATION of the training stream (the first `ntrain`
    records are therefore identical to those of the paper protocol). Exact test∩train leakage reported, removed by default."""
    gen = gmod.GENERATORS[kind]
    off = getattr(gmod, "TEST_SEED_OFFSET", 10_000)
    pool = [x.encode("utf-8") for x in gen(ntrain + nval, seed)]
    train, val = pool[:ntrain], pool[ntrain:]
    test = [x.encode("utf-8") for x in gen(ntest, seed + off)]
    tr = set(train)
    leak = sum(1 for r in test if r in tr) / max(1, len(test))
    if drop_leaked:
        test = [r for r in test if r not in tr]
        val = [r for r in val if r not in tr]
    return train, val, test, leak


def group_concat(recs: list[bytes], k: int) -> list[bytes]:
    return [b"".join(recs[i:i + k]) for i in range(0, len(recs) - k + 1, k)] if k > 1 else recs


def mutate(rec: bytes, p: float, rng: random.Random, alphabet: bytes) -> bytes:
    if p <= 0:
        return rec
    b = bytearray(rec)
    for i in range(len(b)):
        if rng.random() < p:
            b[i] = rng.choice(alphabet)
    return bytes(b)


# ----------------------------------------------------------------------------------------
# Dictionaries (all trained on the SAME training set, same byte budget)
# ----------------------------------------------------------------------------------------

def dict_sampled(train: list[bytes], size: int, seed: int) -> bytes:
    rng = random.Random(seed)
    pool = train[:]
    rng.shuffle(pool)
    out, tot = [], 0
    for r in pool:
        if tot + len(r) > size:
            break
        out.append(r)
        tot += len(r)
    return b"".join(out) if out else pool[0][:size]


def dict_ztrained(train: list[bytes], size: int) -> Optional[bytes]:
    if zstd is None:
        return None
    try:
        return zstd.train_dictionary(size, train, threads=-1).as_bytes()
    except Exception as e:  # Too few samples, etc.
        log(f"[zstd train] Failed for size={size}: {e}")
        return None


def default_dict_builder(train: list[str], budget: int, seed: int = 0) -> set[str]:
    """STAND-IN: greedy selection of frequent factors (df>=3). REPLACE with your
    MDL pipeline (--dict-builder file.py:function; signature (train:list[str], budget:int, seed:int)->set[str])."""
    rng = random.Random(seed)
    sample = train if len(train) <= 2000 else rng.sample(train, 2000)
    df = collections.Counter()
    for r in sample:
        seen = set()
        for L in (6, 10, 16, 24, 40):
            for i in range(len(r) - L + 1):
                seen.add(r[i:i + L])
        df.update(seen)
    cand = sorted(((c - 1) * len(w), w) for w, c in df.items() if c >= 3)
    cand.reverse()
    chosen, tot = [], 0
    for _, w in cand:
        if tot + len(w) > budget:
            continue
        if any(w in c for c in chosen):
            continue
        chosen.append(w)
        tot += len(w)
    return set(chosen)


def load_builder(spec: Optional[str]):
    if not spec:
        return default_dict_builder
    path, func = spec.rsplit(":", 1)
    m = load_module(path, "user_dict_builder")
    return getattr(m, func)


def chunk_words(blob: bytes, size: int = 48) -> set[str]:
    s = blob.decode(LATIN1)
    return {s[i:i + size] for i in range(0, len(s), size) if s[i:i + size]}


def words_to_blob(words) -> bytes:
    return "".join(sorted(words, key=len)).encode(LATIN1)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------
# "Method" Abstraction
# ----------------------------------------------------------------------------------------

@dataclass
class Method:
    family: str
    cfg: dict = field(default_factory=dict)
    comp: Optional[Callable[[bytes], bytes]] = None
    decomp: Optional[Callable[[bytes], bytes]] = None
    size_fn: Optional[Callable[[bytes], int]] = None  # "Estimated" methods (not decodable here)
    dict_bytes: int = 0                               # CHARGED dictionary/header size (derived, see __post_init__)
    raw_dict_bytes: int = 0                           # uncompressed dictionary / header / model size (bytes)
    comp_dict_bytes: int = 0                          # zlib-compressed (stored) size of the same object
    estimate: bool = False
    optimistic: bool = False   # NON-validated estimator, biased in favor of baseline (indicative lower bound)
    repeat: int = 3
    fit_s: float = 0.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        # one rule for every method: charged = raw size (default) or stored/compressed size (--dict-charge zlib)
        if not self.raw_dict_bytes and not self.comp_dict_bytes and self.dict_bytes:
            self.raw_dict_bytes = self.comp_dict_bytes = self.dict_bytes      # legacy call sites
        self.dict_bytes = self.raw_dict_bytes if DICT_CHARGE == "raw" else self.comp_dict_bytes


# ---- Baselines without dictionary (minimal framing when possible) ----

def m_zlib_framed(level=6):
    return Method(f"zlib{level}-framed", comp=lambda x: zlib.compress(x, level), decomp=zlib.decompress,
                  dict_bytes=0, raw_dict_bytes=0)


def m_zlib_raw(level=9, zdict: Optional[bytes] = None, tag=""):
    zd = zdict[-32768:] if zdict else None  # Deflate window = 32 KiB

    def comp(x):
        co = (zlib.compressobj(level, zlib.DEFLATED, -15, 9, zlib.Z_DEFAULT_STRATEGY, zd) if zd
              else zlib.compressobj(level, zlib.DEFLATED, -15, 9))
        return co.compress(x) + co.flush()

    def decomp(y):
        do = zlib.decompressobj(-15, zd) if zd else zlib.decompressobj(-15)
        return do.decompress(y) + do.flush()

    return Method(f"zlib{level}-raw" + (f"+{tag}" if tag else ""), comp=comp, decomp=decomp,
                  **dsz(zd))


def m_zstd(level=19, dict_bytes: Optional[bytes] = None, tag="", auto=False):
    if zstd is None:
        return None
    d = None
    if dict_bytes:
        d = zstd.ZstdCompressionDict(dict_bytes, dict_type=zstd.DICT_TYPE_AUTO if auto else zstd.DICT_TYPE_RAWCONTENT)
    p = zstd.ZstdCompressionParameters.from_level(level, format=zstd.FORMAT_ZSTD1_MAGICLESS, write_content_size=0,
                                                  write_checksum=0, write_dict_id=0)
    cc = zstd.ZstdCompressor(compression_params=p, dict_data=d)
    dc = zstd.ZstdDecompressor(format=zstd.FORMAT_ZSTD1_MAGICLESS, dict_data=d)
    return Method(f"zstd{level}-min" + (f"+{tag}" if tag else ""), comp=cc.compress,
                  decomp=lambda y: dc.decompress(y, max_output_size=1 << 28),
                  **dsz(dict_bytes))


def m_brotli(q=11, lgwin=22):
    if brotli is None:
        return None
    return Method(f"brotli{q}", comp=lambda x: brotli.compress(x, quality=q, lgwin=lgwin, mode=brotli.MODE_GENERIC),
                  decomp=brotli.decompress, dict_bytes=0, raw_dict_bytes=0)


# Reduced dict_size: otherwise each call allocates 64 MiB and distorts per-record throughput (no effect on size)
_LZMA_F = [{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME, "lc": 3, "lp": 0, "pb": 0,
            "dict_size": 1 << 20}]
_LZMA_FS = [{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME, "lc": 3, "lp": 0, "pb": 0,
             "dict_size": 1 << 26}]


def lzma_raw_c(x):
    return lzma.compress(x, format=lzma.FORMAT_RAW, filters=_LZMA_F)


def m_lzma():
    return Method("lzma-raw", comp=lzma_raw_c,
                  decomp=lambda y: lzma.decompress(y, format=lzma.FORMAT_RAW, filters=_LZMA_F),
                  dict_bytes=0, raw_dict_bytes=0)


def m_bz2():
    return Method("bzip2", comp=lambda x: bz2.compress(x, 9), decomp=bz2.decompress, repeat=3,
                  dict_bytes=0, raw_dict_bytes=0)


def ppmd_c(x):
    return pyppmd.compress(x, max_order=8, mem_size=16 << 20)


def primed(family: str, fn: Callable[[bytes], bytes], blob: bytes, tag: str) -> Method:
    """OPTIMISTIC BOUND (not a measurement): marginal cost len(fn(blob+x)) - len(fn(blob)) + len(fn(b"")).
    Used for compressors without a dictionary API in Python (brotli, lzma, ppmd).
    WARNING: Validated on zlib/zstd (ground truth via dictionary API), this estimator is overly
    optimistic by 27% (zlib) to 41% (zstd) even with termination allocation, because entropy tables
    (Huffman trees, FSE, adaptive probas) are 'already warm' instead of being emitted/relearned
    per record. See --validate-estimator. These rows are excluded from the 'codec wins' verdict."""
    base, term = len(fn(blob)), len(fn(b""))
    return Method(f"{family}~primed+{tag}", size_fn=lambda x: len(fn(blob + x)) - base + term, estimate=True,
                  optimistic=True, repeat=1, **dsz(blob))


# ---- Control: trained order-k context model (Witten-Bell), ideal code length ----

class OrderK:
    def __init__(self, k: int, min_n: int):
        self.k, self.min_n = k, min_n

    def fit(self, train: list[bytes]):
        k = self.k
        t = [collections.defaultdict(collections.Counter) for _ in range(k + 1)]
        for r in train:
            pad = b"\x00" * k + r
            for i in range(len(r)):
                for j in range(k + 1):
                    t[j][pad[i + k - j:i + k]][r[i]] += 1
        self.t = []
        buf = bytearray()
        for j in range(k + 1):
            tj = {}
            for ctx, c in t[j].items():
                n = sum(c.values())
                if j >= 1 and n < self.min_n:
                    continue
                tj[ctx] = (c, n, len(c))
                buf += bytes([j]) + ctx + bytes([len(c) & 255])
                for s, cnt in sorted(c.items()):
                    buf += bytes([s]) + vbytes(cnt)
            self.t.append(tj)
        self.model_blob = bytes(buf)

    def bits(self, rec: bytes) -> float:
        k = self.k
        pad = b"\x00" * k + rec
        tot = 0.0
        for i in range(len(rec)):
            p, s = 1 / 256, rec[i]
            for j in range(k + 1):
                e = self.t[j].get(pad[i + k - j:i + k])
                if e is None:
                    break
                c, n, u = e
                p = (c.get(s, 0) + u * p) / (n + u)
            tot -= math.log2(p)
        return tot


def m_orderk(model: OrderK) -> Method:
    return Method(f"ctxmodel-o{model.k}(ideal)", {"min_n": model.min_n},
                  size_fn=lambda x: math.ceil(model.bits(x) / 8) + 1, estimate=True,
                  repeat=1, **dsz(model.model_blob))


# ---- The codec ----

@contextlib.contextmanager
def no_selfref(mod):
    orig = mod.LZ77Index.find_matches
    mod.LZ77Index.find_matches = lambda self, s, pos, max_antecedents=16: []
    try:
        yield
    finally:
        mod.LZ77Index.find_matches = orig


def m_codec(mod, family: str, train: list[bytes], S_words: set[str], refine: int, selfref=True) -> Method:
    t0 = time.perf_counter()
    tr = [b.decode(LATIN1) for b in train]
    cm = contextlib.nullcontext() if selfref else no_selfref(mod)
    with cm:
        # IMPORTANT: alphabet_source = train (the corpus_compressed_size() default would take TEST).
        ctx = mod.build_context(set(S_words), tr, train_corpus=tr, progress=False, refine_passes=refine)
    header = ctx.header_bytes
    ctx_dec, _ = mod.read_context_header(header)  # Decoder reconstructed ONLY from header bytes

    def comp(x: bytes) -> bytes:
        with (contextlib.nullcontext() if selfref else no_selfref(mod)):
            return mod.encode_body(x.decode(LATIN1), ctx)

    def decomp(y: bytes) -> bytes:
        return mod.decode_body(y, ctx_dec).encode(LATIN1)

    # Header = alphabet + dictionary S + probability tables.  RAW = uncompressed header (NOT only the words of S);
    # COMP = the header as stored (zlib).  S_raw is reported separately for information.
    hdr_raw = uncompressed_header_size(mod, ctx)
    hdr_comp = len(header)
    s_raw = len(words_to_blob(S_words))
    return Method(family, comp=comp, decomp=decomp, raw_dict_bytes=hdr_raw, comp_dict_bytes=hdr_comp, repeat=1,
                  fit_s=time.perf_counter() - t0,
                  extra={"S_words": set(S_words), "n_words": len(S_words), "S_raw_bytes": s_raw})


# ----------------------------------------------------------------------------------------
# Candidate Construction, Validation Selection
# ----------------------------------------------------------------------------------------

def standalone_methods(args) -> list[Method]:
    ms = [m_zlib_framed(6), m_zlib_raw(9), m_zstd(19), m_brotli(), m_lzma(), m_bz2()]
    return [m for m in ms if m is not None]


def candidates_for_budget(b, train, seed, mod, builder, args, S_ref=None, include_S=True, cross=True) -> list[Method]:
    """Baseline candidates (+ codec if S is not imposed by MDL) for a dictionary budget b (bytes).
    S_ref: imposed dictionary S (MDL selection); in this case the main codec is built elsewhere."""
    ms: list[Method] = []
    sampled = dict_sampled(train, b, seed)
    t0 = time.perf_counter()
    zt = dict_ztrained(train, b)
    zt_s = time.perf_counter() - t0
    if S_ref is not None:
        S_words = S_ref
    else:
        S_words = builder([x.decode(LATIN1) for x in train], b, seed)
    S_blob = words_to_blob(S_words) if S_words else b""
    sources = {"sampled": (sampled, False)}
    if include_S:
        sources["S"] = (S_blob, False)
    if zt:
        sources["ztrained"] = (zt, True)
    for tag, (blob, auto) in sources.items():
        if not blob:
            continue
        group = []
        group.append(m_zlib_raw(9, blob, tag))
        z = m_zstd(19, blob, tag, auto=auto)
        if z:
            group.append(z)
        if args.primed and tag in ("sampled", "S"):
            if brotli:
                group.append(primed("brotli11", lambda x: brotli.compress(x, quality=11, lgwin=22), blob, tag))
            group.append(primed("lzma", lzma_raw_c, blob, tag))
            if pyppmd:
                group.append(primed("ppmd8", ppmd_c, blob, tag))
        if tag == "ztrained":
            for g in group:
                g.fit_s = zt_s
        ms += group
    if mod is not None and not args.no_codec:
        if S_ref is None and S_words:
            ms.append(m_codec(mod, REF, train, S_words, args.refine))
            ms.append(m_codec(mod, "codec-ablation[RLZ: no self-ref.]", train, S_words, args.refine, selfref=False))
        if zt and cross:
            ms.append(m_codec(mod, "codec-cross[S=zstd-trained]", train, chunk_words(zt), args.refine))
    for m in ms:
        m.cfg["budget"] = b
    return ms


# ----------------------------------------------------------------------------------------
# MDL Selection of S (module mdl_corrige_v10_fast.py)
# ----------------------------------------------------------------------------------------

def mdl_codec_name(args, mode: int) -> str:
    return "CODEC[S=MDL]" if len(args.mdl_modes) == 1 else f"CODEC[S=MDL m={mode}]"


def run_mdl(mdlmod, codec_mod, train: list[bytes], n_total: int, mode: int, seed: int, args):
    """Calls the MDL algorithm ONLY on the training set. λ = n_total/len(train) (number of records to
    compress in production / training records: deployment parameter, no test leakage)
    unless --mdl-lambda; or --mdl-cv: λ chosen by cross-validation internal to train."""
    tr = [x.decode(LATIN1) for x in train]
    if any("\x01" in x for x in tr):
        raise ValueError("MDL assumes character \\x01 (word separator) is absent from data")
    common = dict(max_seq_len=args.mdl_max_seq_len, overlap=args.mdl_overlap, max_seed_size=args.mdl_max_seed,
                  progress=False, n_jobs=args.mdl_jobs, prune_mode=mode, select_search=args.mdl_select_search)
    info: dict = {}
    t0 = time.perf_counter()
    if args.mdl_cv:
        # optimize_dictionary_mdl_cv does not expose codec_module: it imports 'cs_adapt_codec_3_03' by default
        sys.modules.setdefault("cs_adapt_codec", codec_mod)
        S = mdlmod.optimize_dictionary_mdl_cv(tr, n_splits=args.mdl_cv_splits, seed=seed, **common)
    else:
        lam = args.mdl_lambda
        S = mdlmod.optimize_dictionary_mdl(
            tr, n_total=None if lam else n_total, lam=lam, select=args.mdl_select,
            codec_module=codec_mod.__name__, n_checkpoints=args.mdl_checkpoints, info=info, **common)
    return set(S), time.perf_counter() - t0, info


def get_mdl_codecs(train, n_total, seed, mod, mdlmod, args) -> list[Method]:
    """Codecs whose S comes from MDL (one per pruning mode) + ablations on the S of reference mode.
    Cached by (seed, modes): scenario B reuses the same dictionary and the same codec."""
    key = ("mdl", seed, tuple(args.mdl_modes), args.mdl_lambda, args.mdl_cv, args.refine)
    if key in _CACHE:
        return _CACHE[key]
    out: list[Method] = []
    for i, mode in enumerate(args.mdl_modes):
        S, t_mdl, info = run_mdl(mdlmod, mod, train, n_total, mode, seed, args)
        m = m_codec(mod, mdl_codec_name(args, mode), train, S, args.refine)
        m.fit_s += t_mdl
        m.cfg.update(mdl_mode=mode, n_words=len(S))
        m.extra.update(mdl_s=t_mdl, mdl_info=info)
        log(f"  [MDL mode {mode}] |S|={len(S)} words ({len(words_to_blob(S))} B), codec header raw={m.raw_dict_bytes} B "
            f"/ stored(zlib)={m.comp_dict_bytes} B, charged={m.dict_bytes} B, "
            f"MDL {t_mdl:.1f}s (+ context {m.fit_s - t_mdl:.1f}s)" +
            (f", selected checkpoint, |S|={info.get('chosen_size')}" if info else ""))
        out.append(m)
        if i == 0 and not args.no_ablations:
            out.append(m_codec(mod, "codec-ablation[RLZ: no self-ref.]", train, S, args.refine, selfref=False))
            out.append(m_codec(mod, "codec-ablation[S=∅: pure LZ77]", train, set(), args.refine))
    _CACHE[key] = out
    return out


def sizes_of(m: Method, recs: list[bytes]) -> list[int]:
    return [m.size_fn(x) for x in recs] if m.estimate else [len(m.comp(x)) for x in recs]


def total_bytes(sizes: list[int], dict_bytes: int, scale: float = 1.0) -> float:
    return scale * (sum(sizes) + sum(varint_len(s) for s in sizes)) + dict_bytes


def fit_and_select(train, val, ntest_target, seed, mod, builder, args, mdlmod=None) -> list[Method]:
    """Returns, for each family, the config (budget) minimizing EXPECTED total on test,
    estimated on val: total = (payload+index)(val) * ntest/nval + dictionary.
    With MDL, S and codec are built once; baselines also receive MATCHED budget dictionaries
    (size of S and size of codec header)."""
    cands = standalone_methods(args)
    budgets = list(args.budgets)
    S_ref, matched = None, set()
    if mdlmod is not None and mod is not None and not args.no_codec:
        fixed = get_mdl_codecs(train, ntest_target, seed, mod, mdlmod, args)
        cands += fixed
        S_ref = fixed[0].extra["S_words"]
        for x in (len(words_to_blob(S_ref)), fixed[0].dict_bytes):
            if x > 0:
                matched.add(x)
                if x not in budgets:
                    budgets.append(x)
    first = True
    for b in sorted(budgets):
        t0 = time.perf_counter()
        cands += candidates_for_budget(b, train, seed, mod, builder, args, S_ref=S_ref,
                                       include_S=(S_ref is None or first),
                                       cross=(S_ref is None or b in matched))
        first = False
        log(f"  [budget {b}{' (matched)' if b in matched else ''}] candidates ready ({time.perf_counter() - t0:.1f}s)")
    if args.orderk:
        for k in (1, 2, 3):
            for mn in (1, 3):
                o = OrderK(k, mn)
                o.fit(train)
                cands.append(m_orderk(o))
    vsub = val[: args.val_eval_max]
    scale = ntest_target / max(1, len(vsub))
    best: dict[str, tuple[float, Method]] = {}
    for m in cands:
        try:
            tot = total_bytes(sizes_of(m, vsub), m.dict_bytes, scale)
        except Exception as e:
            log(f"  [skip val] {m.family} {m.cfg}: {e}")
            continue
        if m.family not in best or tot < best[m.family][0]:
            best[m.family] = (tot, m)
    return [m for _, m in best.values()]


# ----------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------

def timed(fn, items, reps):
    best, out = INF, None
    for _ in range(reps):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        o = [fn(x) for x in items]
        dt = time.perf_counter() - t0
        gc.enable()
        if dt < best:
            best, out = dt, o
    return best, out


def evaluate(m: Method, recs: list[bytes], check=True) -> dict:
    raw = sum(len(r) for r in recs)
    res = {"family": m.family, "cfg": m.cfg, "dict": m.dict_bytes, "raw_dict": m.raw_dict_bytes,
           "comp_dict": m.comp_dict_bytes, "s_raw": m.extra.get("S_raw_bytes"),
           "estimate": m.estimate, "optimistic": m.optimistic, "n": len(recs), "raw": raw, "fit_s": m.fit_s}
    if m.estimate:
        sizes = [m.size_fn(x) for x in recs]
        res.update(ok=None, enc_mbs=None, dec_mbs=None, enc_us=None, dec_us=None)
    else:
        te, outs = timed(m.comp, recs, m.repeat)
        sizes = [len(o) for o in outs]
        ok = None
        td = None
        if m.decomp is not None:
            td, decs = timed(m.decomp, outs, m.repeat)
            ok = all(d == x for d, x in zip(decs, recs)) if check else None
        res.update(ok=ok, enc_mbs=raw / te / 1e6, enc_us=te / len(recs) * 1e6,
                   dec_mbs=(raw / td / 1e6) if td else None, dec_us=(td / len(recs) * 1e6) if td else None)
    payload, index = sum(sizes), sum(varint_len(s) for s in sizes)
    total = payload + index + m.dict_bytes
    total_raw = payload + index + m.raw_dict_bytes
    total_comp = payload + index + m.comp_dict_bytes
    res.update(sizes=sizes, payload=payload, index=index, total=total, total_raw=total_raw, total_comp=total_comp,
               ratio=total / raw, ratio_raw=total_raw / raw, ratio_comp=total_comp / raw)
    return res


def boot_gain(a: list[int], b: list[int], B=1000, seed=0) -> tuple[float, float]:
    """Paired bootstrap 95% CI of 1 - sum(a)/sum(b) (payload only, excluding dictionary)."""
    n = len(a)
    if n == 0:
        return (float("nan"),) * 2
    rng = random.Random(seed)
    if np is not None:
        A, Bv = np.asarray(a, float), np.asarray(b, float)
        idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
        g = 1 - A[idx].sum(1) / Bv[idx].sum(1)
        return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))
    gs = []
    for _ in range(min(B, 300)):
        ix = [rng.randrange(n) for _ in range(n)]
        gs.append(1 - sum(a[i] for i in ix) / sum(b[i] for i in ix))
    gs.sort()
    return gs[int(0.025 * len(gs))], gs[int(0.975 * len(gs)) - 1]


def aggregate(runs: list[dict[str, dict]], ref=None) -> list[dict]:
    ref = ref or REF
    fams = {f for r in runs for f in r}
    rows = []
    for f in fams:
        rs = [r[f] for r in runs if f in r]
        row = {"family": f, "seeds": len(rs),
               "cfg": collections.Counter(json.dumps(x["cfg"], sort_keys=True) for x in rs).most_common(1)[0][0],
               "payload": statistics.mean(x["payload"] for x in rs),
               "dict": statistics.mean(x["dict"] for x in rs),
               "raw_dict": statistics.mean(x.get("raw_dict", x["dict"]) for x in rs),
               "comp_dict": statistics.mean(x.get("comp_dict", x["dict"]) for x in rs),
               "index": statistics.mean(x["index"] for x in rs),
               "n_rec": statistics.mean(x["n"] for x in rs),
               "total": statistics.mean(x["total"] for x in rs),
               "total_raw": statistics.mean(x.get("total_raw", x["total"]) for x in rs),
               "total_comp": statistics.mean(x.get("total_comp", x["total"]) for x in rs),
               "ratio%": 100 * statistics.mean(x["ratio"] for x in rs),
               "ratio_raw%": 100 * statistics.mean(x.get("ratio_raw", x["ratio"]) for x in rs),
               "std_total": statistics.pstdev([x["total"] for x in rs]) if len(rs) > 1 else 0.0,
               "estimate": rs[0]["estimate"], "optimistic": rs[0].get("optimistic", False)}
        row["fit_s"] = statistics.mean(x.get("fit_s", 0.0) for x in rs)
        for key in ("enc_mbs", "dec_mbs", "enc_us", "dec_us"):
            vals = [x[key] for x in rs if x.get(key)]
            row[key] = statistics.median(vals) if vals else None
        oks = [x["ok"] for x in rs if x["ok"] is not None]
        row["roundtrip"] = "—" if not oks else ("OK" if all(oks) else "FAIL")
        if ref in {k for r in runs for k in r} and f != ref:
            gains, gains_raw, gains_comp, wins, A, Bv = [], [], [], 0, [], []
            for r in runs:
                if ref in r and f in r:
                    g = 1 - r[ref]["total"] / r[f]["total"]
                    g_raw = 1 - r[ref].get("total_raw", r[ref]["total"]) / r[f].get("total_raw", r[f]["total"])
                    gains.append(g)
                    gains_raw.append(g_raw)
                    gains_comp.append(1 - r[ref].get("total_comp", r[ref]["total"]) / r[f].get("total_comp", r[f]["total"]))
                    wins += g > 0
                    A += r[ref]["sizes"]
                    Bv += r[f]["sizes"]
            row["gain%"] = 100 * statistics.mean(gains)
            row["gain_raw%"] = 100 * statistics.mean(gains_raw)
            row["gain_comp%"] = 100 * statistics.mean(gains_comp)
            row["gain_std"] = 100 * (statistics.pstdev(gains) if len(gains) > 1 else 0.0)
            row["wins"] = f"{wins}/{len(gains)}"
            lo, hi = boot_gain(A, Bv)
            row["ci_payload%"] = (100 * lo, 100 * hi)
            # Break-even point (number of records beyond which codec becomes better, dictionaries included)
            dg = statistics.mean(
                (sum(r[f]["sizes"]) - sum(r[ref]["sizes"])) / r[ref]["n"] for r in runs if ref in r and f in r)
            dd = row_ref_dict(runs, ref) - row["dict"]
            row["breakeven_n"] = (dd / dg) if dg > 0 and dd > 0 else (0 if dd <= 0 and dg > 0 else None)
        rows.append(row)
    return sorted(rows, key=lambda r: r["total"])


def row_ref_dict(runs, ref):
    return statistics.mean(r[ref]["dict"] for r in runs if ref in r)


def fmt(v, spec=".1f"):
    return "—" if v is None else format(v, spec)


def md_table(rows: list[dict], title: str, ref: Optional[str] = None) -> str:
    ref = ref or REF
    main = [r for r in rows if not r.get("optimistic")]
    opt = [r for r in rows if r.get("optimistic")]
    txt = _md_rows(main, title, ref)
    if opt:
        txt += ("\n> ⚠ **Optimistic bounds, NOT comparable**: marginal cost with history (brotli/lzma/ppmd have no "
                "dictionary API in Python). Validated on zlib/zstd, the estimator underestimates actual cost by 27 to 41%. "
                "They should not be read as measurements.\n\n")
        txt += _md_rows(opt, title + " — optimistic bounds", ref)
    return txt


def _md_rows(rows: list[dict], title: str, ref: str) -> str:
    h = ("| Method | cfg(val) | payload | dict (comp) | dict (raw) | index | **total** | **total (raw)** | ratio % | ratio raw % | codec gain % | gain raw % | "
         "95% CI payload % | codec wins | break-even n | offline s | enc MB/s | dec MB/s | dec µs/rec. | RT |\n"
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--|\n")
    out = [f"### {title}\n", h]
    for r in rows:
        ci = r.get("ci_payload%")
        gain_str = ("ref." if r["family"] == ref else "—") if "gain%" not in r else f"{r['gain%']:+.1f} ± {r['gain_std']:.1f}"
        gain_raw_str = ("ref." if r["family"] == ref else "—") if "gain_raw%" not in r else f"{r['gain_raw%']:+.1f}"
        out.append("| {}{} | {} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | **{:.0f}** | **{:.0f}** | {:.2f} | {:.2f} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
            r["family"], " *(est.)*" if r["estimate"] else "", r["cfg"].replace("|", "/"), r["payload"], r["dict"], r["raw_dict"],
            r["index"], r["total"], r["total_raw"], r["ratio%"], r["ratio_raw%"],
            gain_str, gain_raw_str,
            "—" if not ci else f"[{ci[0]:+.1f}, {ci[1]:+.1f}]", r.get("wins", "—"),
            "—" if r.get("breakeven_n") is None else f"{r['breakeven_n']:.0f}",
            fmt(r["fit_s"] if r.get("fit_s", 0) > 0.05 else None, ".1f"), fmt(r["enc_mbs"], ".3f"), fmt(r["dec_mbs"], ".3f"), fmt(r["dec_us"], ".0f"), r["roundtrip"]))
    return "".join(out)


PRETTY_NOTE = "payload = sum of compressed record sizes; index = one varint length per record"


def tex_escape(x: str) -> str:
    return x.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def tex_table(rows: list[dict], caption: str, meta: Optional[dict] = None) -> str:
    """One row per method. Columns (parsed by make_experiment_tables.py):
    Method & payload & index & dict.(raw) & dict.(comp) & total (charged) & ratio (charged) & gain (charged) & gain std.
    The first line is a machine-readable '% bench-meta' comment (charging rule, n_test, mean record length...)."""
    m = " ".join(f"{k}={v}" for k, v in (meta or {}).items())
    s = f"% bench-meta: {m}\n" + (
         "\\begin{table}[h]\\centering\\footnotesize\\caption{%s}\\begin{tabular}{lrrrrrrrr}\\toprule\n"
         "Method & payload & index & dict.(raw) & dict.(comp) & total (B) & ratio (\\%%) & codec gain (\\%%) & gain std \\\\\\midrule\n" % caption)
    for r in rows:
        s += "%s & %.0f & %.0f & %.0f & %.0f & %.0f & %.2f & %s & %s \\\\\n" % (
            tex_escape(r["family"]), r["payload"], r["index"], r["raw_dict"], r["comp_dict"], r["total"], r["ratio%"],
            "--" if "gain%" not in r else f"{r['gain%']:+.2f}", "--" if "gain_std" not in r else f"{r['gain_std']:.2f}")
    return s + "\\bottomrule\\end{tabular}\\end{table}\n"


# ----------------------------------------------------------------------------------------
# Scenario S: "Solid" baselines (loss of random access) incl. lrzip
# ----------------------------------------------------------------------------------------

def cli_solid(name, comp_cmd, dec_cmd):
    exe = comp_cmd.split()[0]
    if shutil.which(exe) is None:
        return None

    def comp(x: bytes) -> bytes:
        with tempfile.TemporaryDirectory() as d:
            i, o = os.path.join(d, "in"), os.path.join(d, "out")
            open(i, "wb").write(x)
            subprocess.run(comp_cmd.format(i=i, o=o).split(), check=True, capture_output=True)
            return open(o, "rb").read()

    def decomp(y: bytes) -> bytes:
        with tempfile.TemporaryDirectory() as d:
            i, o = os.path.join(d, "in"), os.path.join(d, "out")
            open(i, "wb").write(y)
            subprocess.run(dec_cmd.format(i=i, o=o).split(), check=True, capture_output=True)
            return open(o, "rb").read()

    return Method(name, comp=comp, decomp=decomp, repeat=1, dict_bytes=0, raw_dict_bytes=0)


def solid_methods(extra_cli: list[str]) -> list[Method]:
    ms = []
    if zstd:
        p = zstd.ZstdCompressionParameters.from_level(22, window_log=27, enable_ldm=1, write_content_size=0,
                                                      write_checksum=0)
        cc, dc = zstd.ZstdCompressor(compression_params=p), zstd.ZstdDecompressor(max_window_size=1 << 28)
        ms.append(Method("SOLID zstd22+ldm", comp=cc.compress, decomp=lambda y: dc.decompress(y, max_output_size=1 << 30),
                         repeat=1, dict_bytes=0, raw_dict_bytes=0))
    if brotli:
        ms.append(Method("SOLID brotli11/lgwin24", comp=lambda x: brotli.compress(x, quality=11, lgwin=24),
                         decomp=brotli.decompress, repeat=1, dict_bytes=0, raw_dict_bytes=0))
    ms.append(Method("SOLID xz-raw lzma1 9e", comp=lambda x: lzma.compress(x, format=lzma.FORMAT_RAW, filters=_LZMA_FS),
                     decomp=lambda y: lzma.decompress(y, format=lzma.FORMAT_RAW, filters=_LZMA_FS), repeat=1,
                     dict_bytes=0, raw_dict_bytes=0))
    if pyppmd:
        ms.append(Method("SOLID ppmd8 o16", comp=lambda x: pyppmd.compress(x, max_order=16, mem_size=256 << 20),
                         decomp=lambda y: pyppmd.decompress(y, max_order=16, mem_size=256 << 20), repeat=1,
                         dict_bytes=0, raw_dict_bytes=0))
    ms.append(Method("SOLID bzip2-9", comp=lambda x: bz2.compress(x, 9), decomp=bz2.decompress, repeat=1,
                     dict_bytes=0, raw_dict_bytes=0))
    for name, c, d in [("SOLID lrzip (rzip+lzma)", "lrzip -q -f -L 9 -o {o} {i}", "lrzip -q -f -d -o {o} {i}"),
                       ("SOLID lrzip -z (zpaq)", "lrzip -q -f -z -L 9 -o {o} {i}", "lrzip -q -f -d -o {o} {i}"),
                       ("SOLID lrzip -b (bzip2)", "lrzip -q -f -b -L 9 -o {o} {i}", "lrzip -q -f -d -o {o} {i}")]:
        m = cli_solid(name, c, d)
        if m:
            ms.append(m)
    for spec in extra_cli:  # "name|comp cmd with {i} {o}|decomp cmd"
        n, c, d = spec.split("|")
        m = cli_solid("SOLID " + n, c, d)
        if m:
            ms.append(m)
    return ms


def run_solid(selected: list[Method], test: list[bytes], extra_cli) -> list[dict]:
    blob = b"".join(test)
    idx = sum(varint_len(len(r)) for r in test)
    out = []
    for m in solid_methods(extra_cli) + [s for s in selected if s.family in (
            REF, "zstd19-min+S", "zstd19-min+ztrained", "zlib9-raw+S")]:
        try:
            r = evaluate(m, [blob])
        except Exception as e:
            log(f"[solid] {m.family}: {e}")
            continue
        r["family"] = ("(in block) " + m.family) if not m.family.startswith("SOLID") else m.family
        r["index"] = idx
        r["total"] = r["payload"] + idx + m.dict_bytes
        r["total_raw"] = r["payload"] + idx + m.raw_dict_bytes
        r["ratio"] = r["total"] / r["raw"]
        r["ratio_raw"] = r["total_raw"] / r["raw"]
        r["sizes"] = [r["payload"]]
        out.append(r)
    return out


# ----------------------------------------------------------------------------------------
# Scenario T: Empirical verifications of the paper's theoretical results
# ----------------------------------------------------------------------------------------

def greedy_count(mod, s: str, sam_S, use_P=True, ctx="") -> int:
    sam_P = mod.SuffixAutomaton() if use_P else None
    if use_P:
        for c in ctx:
            sam_P.extend(c)
    i, k, n = 0, 0, len(s)
    while i < n:
        lS = len(sam_S.all_matches(s, i)) if sam_S is not None else 0
        lP = len(sam_P.all_matches(s, i)) if sam_P is not None else 0
        l = max(1, lS, lP)
        if use_P:
            for c in s[i:i + l]:
                sam_P.extend(c)
        i += l
        k += 1
    return k


def exact_adaptive(s: str, S_list: list[str], ctx="") -> int:
    """Exact DP optimum (block admissibility depends only on its start position)."""
    n = len(s)
    dp = [10 ** 9] * (n + 1)
    dp[0] = 0
    for i in range(n):
        hist = ctx + s[:i]
        for j in range(i + 1, n + 1):
            b = s[i:j]
            if j == i + 1 or b in hist or any(b in w for w in S_list):
                dp[j] = min(dp[j], dp[i] + 1)
            else:
                break
    return dp[n]


def theory_checks(mod, S_words: set[str], test: list[bytes], n_random=300, seed=0) -> dict:
    rng = random.Random(seed)
    res = collections.Counter()
    # (1) random strings: greedy optimality, log law, domination, preset-history sandwich
    for _ in range(n_random):
        sig = "ab" if rng.random() < 0.5 else "abc"
        S_list = ["".join(rng.choice(sig) for _ in range(rng.randint(2, 8))) for _ in range(rng.randint(0, 4))]
        s = "".join(rng.choice(sig) for _ in range(rng.randint(5, 40)))
        sam_S = mod.build_dict_automaton(S_list) if S_list else None
        g = greedy_count(mod, s, sam_S)
        res["greedy==exact_violations"] += g != exact_adaptive(s, S_list)
        cs = greedy_count(mod, s, sam_S, use_P=False)
        res["log_law_violations"] += not (1 + math.ceil(math.log2(cs)) <= g <= cs)
        c0 = greedy_count(mod, s, None)
        res["monotonicity_violations"] += not (g <= c0)
        R = "".join(rng.choice(sig) for _ in range(rng.randint(2, 12)))
        ctxc = exact_adaptive(s, [], ctx=R)
        cR = exact_adaptive(s, [R])
        res["sandwich_violations"] += not (ctxc <= cR <= 2 * ctxc)
        res["random_cases"] += 1
    for n in (1, 2, 3, 5, 8, 17, 64, 100):  # tightness a^n
        res["a^n_tightness_violations"] += greedy_count(mod, "a" * n, None) != 1 + math.ceil(math.log2(n))
    # (2) on real records
    S_list = sorted(S_words)
    sam_S = mod.build_dict_automaton(S_list) if S_list else None
    etas, ca, cs_, c0_ = [], [], [], []
    for x in test[:200]:
        s = x.decode(LATIN1)
        g = greedy_count(mod, s, sam_S)
        cs = greedy_count(mod, s, sam_S, use_P=False)
        c0 = greedy_count(mod, s, None)
        res["real_log_law_violations"] += not (1 + math.ceil(math.log2(cs)) <= g <= cs)
        res["real_domination_violations"] += not (g <= c0)
        res["real_records"] += 1
        ca.append(g), cs_.append(cs), c0_.append(c0), etas.append(c0 / g)
    out = dict(res)
    if ca:
        out.update(mean_C_adapt=statistics.mean(ca), mean_C_static=statistics.mean(cs_),
                   mean_C_empty=statistics.mean(c0_), mean_eta=statistics.mean(etas),
                   mean_len=statistics.mean(len(x) for x in test[:200]))
    return out


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------

def validate_estimator(train, test, budget=2048, seed=0):
    """Compares the marginal estimator ('primed') to ground truth on compressors that have a real
    dictionary API (zlib, zstd). A strongly negative bias = overly optimistic estimator."""
    blob = dict_sampled(train, budget, seed)

    def zl(x):
        co = zlib.compressobj(9, zlib.DEFLATED, -15, 9)
        return co.compress(x) + co.flush()

    def zs(x):
        p = zstd.ZstdCompressionParameters.from_level(19, format=zstd.FORMAT_ZSTD1_MAGICLESS, write_content_size=0,
                                                      write_checksum=0, write_dict_id=0)
        return zstd.ZstdCompressor(compression_params=p).compress(x)

    out = []
    for name, fn, real in [("zlib9", zl, m_zlib_raw(9, blob, "s")), ("zstd19", zs, m_zstd(19, blob, "s"))]:
        if real is None:
            continue
        est = primed(name, fn, blob, "s")
        r = sum(len(real.comp(x)) for x in test)
        e = sum(est.size_fn(x) for x in test)
        out.append((name, r, e, 100 * (e - r) / r))
        print(f"{name}: real payload={r}  estimated(marginal+termination)={e}  bias={100 * (e - r) / r:+.1f}%  "
              f"({(r - e) / len(test):.1f} B/rec. underestimated)")
    return out


# ----------------------------------------------------------------------------------------
# --nsweep: Total cost as a function of the number of compressed records (dictionary amortization)
# ----------------------------------------------------------------------------------------

def pretty(name: str) -> str:
    """Human-readable method labels (figure legends)."""
    m = re.match(r"ctxmodel-o(\d)", name)
    if m:
        return f"order-{m.group(1)} context model (ideal)"
    if name.startswith("CODEC[S=MDL"):
        mm = re.search(r"m=(-?\d)", name)
        return {"-1": "Codec (MDL, exact pruning)", "1": "Codec (MDL, lazy pruning)", "0": "Codec (MDL, dead-only)",
                "2": "Codec (MDL, lazy-budget)"}.get(mm.group(1) if mm else "", name)
    if "RLZ" in name:
        return "Codec w/o self-reference (RLZ)"
    if "S=∅" in name or "S=\u2205" in name:
        return "Codec w/o dictionary (S=∅)"
    if name.startswith("codec-cross") or "croisé" in name:
        return "Codec, S = zstd-trained chunks"
    src = {"ztrained": "trained dict.", "sampled": "sampled dict.", "S": "MDL dict. S"}
    m = re.match(r"(zlib9-raw|zstd19-min)\+(\w+)$", name)
    if m:
        return ("zlib-9" if m.group(1).startswith("zlib") else "zstd-19") + " + " + src.get(m.group(2), m.group(2))
    plain = {"brotli11": "Brotli-11", "zstd19-min": "zstd-19", "zlib9-raw": "zlib-9 (raw deflate)",
             "zlib6-framed": "zlib-6", "lzma-raw": "LZMA (raw)", "bzip2": "bzip2"}
    return plain.get(name, name)


def nsweep(runs: list[dict[str, dict]], ns: list[int], ref: Optional[str] = None,
           floor_bytes: Optional[float] = None) -> list[dict]:
    """Model: total(n) = dictionary + n * c, where c = average cost per record (payload + index) measured on test,
    averaged over seeds. For n <= test size this is the measured expectation; beyond it is an EXTRAPOLATION
    with frozen dictionary (neither codec nor baselines are re-trained with more data).
    The dictionary term is the CHARGED one (raw by default, same rule for all methods); raw and stored sizes
    are exported too. Optimistic bounds (~primed) are excluded."""
    ref = ref or REF
    fams = {f for r in runs for f, x in r.items() if not x.get("optimistic")}
    st = {}
    for f in fams:
        xs = [r[f] for r in runs if f in r]
        st[f] = dict(dict=statistics.mean(x["dict"] for x in xs),
                     raw_dict=statistics.mean(x.get("raw_dict", x["dict"]) for x in xs),
                     comp_dict=statistics.mean(x.get("comp_dict", x["dict"]) for x in xs),
                     slope=statistics.mean(sum(z + varint_len(z) for z in x["sizes"]) / len(x["sizes"]) for x in xs),
                     payload_pr=statistics.mean(sum(x["sizes"]) / len(x["sizes"]) for x in xs),
                     rawlen=statistics.mean(x["raw"] / x["n"] for x in xs))
    rows = []
    for f, a in st.items():
        row = {"family": f, "dict": a["dict"], "raw_dict": a["raw_dict"], "comp_dict": a["comp_dict"], "slope": a["slope"],
               "payload_pr": a["payload_pr"], "rawlen": a["rawlen"],
               "ratio_inf%": 100 * a["slope"] / a["rawlen"],
               "per_n": {n: a["dict"] / n + a["slope"] for n in ns},
               "per_n_raw": {n: a["raw_dict"] / n + a["slope"] for n in ns},
               "per_n_comp": {n: a["comp_dict"] / n + a["slope"] for n in ns}, "verdict": ""}
        if ref in st and f != ref:
            r_ = st[ref]
            dd, dm = r_["dict"] - a["dict"], a["slope"] - r_["slope"]   # ref better <=> dd < n*dm
            if dm > 0:
                row["verdict"] = "ref always better" if dd <= 0 else f"ref better for n > {dd / dm:,.0f}"
            elif dm < 0:
                row["verdict"] = f"ref better for n < {dd / dm:,.0f}" if dd < 0 else "ref never better"
            else:
                row["verdict"] = "ref always better" if dd < 0 else "ref never better"
        elif f == ref:
            row["verdict"] = "ref."
        rows.append(row)
    if floor_bytes is not None:     # Shannon floor: no dictionary, no index
        rl = statistics.mean(st[f]["rawlen"] for f in st)
        rows.append({"family": "Shannon FLOOR (source, no index)", "dict": 0.0, "raw_dict": 0.0, "comp_dict": 0.0,
                     "slope": floor_bytes, "payload_pr": floor_bytes, "rawlen": rl,
                     "ratio_inf%": 100 * floor_bytes / rl, "per_n": {n: floor_bytes for n in ns},
                     "per_n_raw": {n: floor_bytes for n in ns}, "per_n_comp": {n: floor_bytes for n in ns},
                     "verdict": "lower bound"})
    return rows


def nsweep_md(rows: list[dict], ns: list[int], n_test: int, title: str) -> str:
    ns_sorted = sorted(ns)
    pivot = min(ns_sorted, key=lambda n: abs(n - n_test))
    rows = sorted(rows, key=lambda r: r["per_n"][pivot])
    best = {n: min(r["per_n"][n] for r in rows) for n in ns_sorted}
    t = [f"### {title}\n\n",
         f"Amortized bytes per record (payload + index + charged dictionary/n; charge = {DICT_CHARGE}). "
         f"n ≤ {n_test}: measured; beyond: extrapolation with frozen dictionary. **Bold** = best at this n. "
         f"\"slope\" = asymptote (n → ∞).\n\n",
         "| Method | dict charged | dict (raw) | dict (comp) | " + " | ".join(f"n={n:,}" for n in ns_sorted) + " | slope (B/rec.) | ratio ∞ % | vs ref. |\n",
         "|---|--:|--:|--:|" + "--:|" * len(ns_sorted) + "--:|--:|---|\n"]
    for r in rows:
        cells = []
        for n in ns_sorted:
            v = r["per_n"][n]
            cells.append(f"**{v:.1f}**" if abs(v - best[n]) < 1e-9 else f"{v:.1f}")
        t.append(f"| {r['family']} | {r['dict']:.0f} | {r['raw_dict']:.0f} | {r['comp_dict']:.0f} | " + " | ".join(cells) +
                 f" | {r['slope']:.2f} | {r['ratio_inf%']:.2f} | {r['verdict']} |\n")
    return "".join(t)


def nsweep_files(rows, ns, base, ref, n_test=None, label=None):
    import csv
    ns_sorted = sorted(ns)
    with open(base + "_nsweep.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "dict_bytes", "dict_bytes_raw", "dict_bytes_comp", "slope_bytes_per_record",
                    "ratio_inf_pct", "payload_bytes_per_record", "charge", "n_test"] +
                   [f"bytes_per_record_n{n}" for n in ns_sorted] + ["vs_ref"])
        for r in rows:
            w.writerow([r["family"], f"{r['dict']:.1f}", f"{r['raw_dict']:.1f}", f"{r['comp_dict']:.1f}",
                        f"{r['slope']:.3f}", f"{r['ratio_inf%']:.3f}", f"{r['payload_pr']:.3f}", DICT_CHARGE, n_test or ""] +
                       [f"{r['per_n'][n]:.3f}" for n in ns_sorted] + [r["verdict"]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np
    except ImportError:
        log("[nsweep] matplotlib missing: no PNG")
        return
    # Same curves as the paper's amortization table: codec, best dictionary baseline, best idealized control
    # (selected at n = n_test), plus the lazy-pruning variant and the Shannon floor when available.
    piv = min(ns_sorted, key=lambda n: abs(n - (n_test or 1000)))
    by = lambda pred: min((r for r in rows if pred(r["family"])), key=lambda r: r["per_n"][piv], default=None)
    is_dict = lambda f: bool(re.match(r"(zlib9-raw|zstd19-min)\+\w+$", f))
    is_ctl = lambda f: f.startswith("ctxmodel-")
    picks = [r for r in rows if r["family"] == ref]
    picks += [r for r in rows if r["family"].startswith("CODEC[S=MDL") and r["family"] != ref][:1]
    for r in (by(is_dict), by(lambda f: is_dict(f) and f.startswith("zstd")), by(is_ctl),
              by(lambda f: f in ("brotli11",))):
        if r is not None and r not in picks:
            picks.append(r)
    fl = [r for r in rows if r["family"].startswith("Shannon")]
    grid = _np.logspace(math.log10(min(ns_sorted)), math.log10(max(ns_sorted)), 300)
    styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, r in enumerate(picks):
        ax.plot(grid, r["dict"] / grid + r["slope"], lw=3.0 if r["family"] == ref else 1.8,
                ls="-" if r["family"] == ref else styles[1 + (i % 5)], color="k" if r["family"] == ref else None,
                label=pretty(r["family"]))
    if fl:
        ax.axhline(fl[0]["slope"], color="gray", lw=1.2, ls=(0, (1, 2)), label="Shannon floor (source)")
    if n_test:
        ax.axvline(n_test, color="gray", lw=0.8, alpha=.6)
        ax.annotate(f"$n_{{test}}$={n_test}", (n_test, ax.get_ylim()[0]), rotation=90, va="bottom", ha="right",
                    fontsize=9, color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of compressed records $n$")
    ax.set_ylabel("Amortized cost (bytes / record)")
    ax.set_title(label or "", fontsize=11)
    ax.grid(True, which="both", alpha=.3)
    ax.legend(fontsize=9, loc="upper right", framealpha=.9)
    fig.tight_layout()
    fig.savefig(base + "_nsweep.png", dpi=200)
    plt.close(fig)


def corpus_name(args) -> str:
    if args.name:
        name = args.name
    elif args.gen_module:
        name = f"gen-{args.gen_kind}"
        if args.gen_reps:
            name += f"_reps{args.gen_reps.replace(',', '-')}"
        if args.gen_motifs:
            name += f"_m{args.gen_motifs}"
    elif args.synthetic:
        name = f"synthetic-{args.synthetic}"
    else:
        path = args.corpus.rstrip("/\\")
        name = os.path.basename(path)
        if os.path.isfile(path):
            name = os.path.splitext(name)[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "corpus"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="file (1 record/line or FASTA) or folder (1 record/file)")
    ap.add_argument("--format", default="lines", choices=["lines", "fasta", "csv"],
                    help="lines: 1 record/line; fasta; csv: 1 record = 1 CSV column (see --csv-column)")
    ap.add_argument("--csv-column", help="CSV: take only the RAW value of this column (name or 0-based index). "
                    "Without this option, each row becomes a JSON object {column: value}")
    ap.add_argument("--csv-columns", help="CSV -> JSON: keep only these columns (comma-separated list)")
    ap.add_argument("--csv-drop", help="CSV -> JSON: drop these columns (e.g. LineId,EventId,EventTemplate)")
    ap.add_argument("--csv-infer-types", action="store_true", help="CSV -> JSON: integers (without leading zero) become JSON numbers")
    ap.add_argument("--json-compact", action="store_true", help="CSV -> JSON: compact serialization (',' and ':' without spaces)")
    ap.add_argument("--csv-delimiter", default=",", help="column delimiter (',' ';' or '\\t')")
    ap.add_argument("--csv-no-header", action="store_true", help="CSV has no header line")
    ap.add_argument("--csv-encoding", default="utf-8")
    ap.add_argument("--fasta-headers", action="store_true")
    ap.add_argument("--crop", type=int, default=0, help="truncates each record to N bytes")
    ap.add_argument("--synthetic", choices=["json", "dna"], help="TOY data for testing the script (do not publish)")
    ap.add_argument("--gen-module", help="generator module (corpora_1.py / corpora_2.py): paper protocol "
                    "(train=gen(seed), test=gen(seed+offset)); replaces --corpus/--synthetic")
    ap.add_argument("--gen-kind", help="generator kind from module: genomic | json | ...")
    ap.add_argument("--gen-reps", help="corpora_2: 'lo,hi' range for genomic motif repeats (default 4,100)")
    ap.add_argument("--gen-motifs", type=int, help="corpora_2: motif pool size (default 100)")
    ap.add_argument("--group-regex", help="group regex (e.g. 'ENSG\\d+'): a group will not cross train/test")
    ap.add_argument("--ntrain", type=int, default=200)
    ap.add_argument("--nval", type=int, default=100)
    ap.add_argument("--ntest", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--budgets", default="512,2048,8192", help="dictionary budgets (bytes), swept for ALL")
    ap.add_argument("--val-eval-max", type=int, default=150)
    ap.add_argument("--scenarios", default="A,T", help="A,B,C,S,T")
    ap.add_argument("--codec", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cs_adapt_codec.py"))
    ap.add_argument("--dict-builder", help="file.py:function (your MDL selection)")
    ap.add_argument("--refine", type=int, default=3)
    g = ap.add_argument_group("MDL selection of S (mdl.py)")
    g.add_argument("--mdl", help="path to mdl_corrige_v10_fast.py: S is then selected by MDL (replaces --dict-builder)")
    g.add_argument("--mdl-prune", default="exact", help="pruning mode(s): exact(-1) | dead(0) | lazy(1) | lazybudget(2), "
                   "comma-separated; the 1st is reference (e.g. exact,lazy)")
    g.add_argument("--mdl-select", default="codec", choices=["codec", "none"], help="stopping point selection by actual codec cost")
    g.add_argument("--mdl-select-search", default="grid", choices=["grid", "refine"])
    g.add_argument("--mdl-checkpoints", type=int, default=25)
    g.add_argument("--mdl-lambda", type=float, default=None, help="fixed λ (default: n_test/n_train)")
    g.add_argument("--mdl-cv", action="store_true", help="select λ by cross-validation internal to train")
    g.add_argument("--mdl-cv-splits", type=int, default=8)
    g.add_argument("--mdl-jobs", type=int, default=None, help="MDL processes (None = all cores)")
    g.add_argument("--mdl-max-seq-len", type=int, default=2000)
    g.add_argument("--mdl-overlap", type=int, default=100)
    g.add_argument("--mdl-max-seed", type=int, default=None)
    ap.add_argument("--validate-estimator", action="store_true",
                    help="measures bias of 'primed' estimator against zlib/zstd (ground truth), then exits")
    g.add_argument("--no-ablations", action="store_true", help="no RLZ / S=∅ codec ablations")
    ap.add_argument("--no-codec", action="store_true")
    ap.add_argument("--dict-charge", default="raw", choices=["zlib", "raw"],
                    help="charging rule applied to EVERY method alike: raw (default) = uncompressed size "
                         "(codec: whole uncompressed header = alphabet + S + tables; baselines: raw dictionary / "
                         "model); zlib = stored size (zlib-compressed). Both sizes are always reported.")
    ap.add_argument("--no-primed", action="store_true")
    ap.add_argument("--no-orderk", action="store_true")
    ap.add_argument("--keep-leaked", action="store_true", help="do not remove test records present in train")
    ap.add_argument("--ks", default="1,4,16", help="scenario B: number of concatenated records per 'record'")
    ap.add_argument("--drift", default="0,0.01,0.05,0.2", help="scenario C: substitution rate")
    ap.add_argument("--extra-cli", action="append", default=[], help="'name|comp cmd {i} {o}|decomp cmd {i} {o}' (scenario S)")
    ap.add_argument("--out", default="bench_out")
    ap.add_argument("--name", help="corpus name in output files (default: file/folder name, or synthetic-<type>)")
    ap.add_argument("--nsweep", action="store_true",
                    help="cost per record as a function of record count n (dictionary amortization), from scenario A")
    ap.add_argument("--title", default="", help="title drawn on the n-sweep figure (default: none; the LaTeX subcaption names the panel)")
    ap.add_argument("--nsweep-n", default="10,100,1000,10000,100000,1000000", help="n values for --nsweep")
    args = ap.parse_args()
    global DICT_CHARGE
    DICT_CHARGE = args.dict_charge
    args.budgets = [int(x) for x in args.budgets.split(",")]
    args.primed, args.orderk = not args.no_primed, not args.no_orderk
    scen = set(args.scenarios.upper().split(","))
    os.makedirs(args.out, exist_ok=True)

    gmod, recs, floor_bits = None, None, None
    if args.gen_module:
        if not args.gen_kind:
            ap.error("--gen-module requires --gen-kind")
        if args.gen_reps:
            os.environ["CORPORA_GENOMIC_REPS"] = args.gen_reps
        if args.gen_motifs:
            os.environ["CORPORA_N_MOTIFS"] = str(args.gen_motifs)
        gmod = load_module(args.gen_module, "corpora_gen")
        if args.gen_kind not in gmod.GENERATORS:
            ap.error(f"--gen-kind {args.gen_kind!r} unknown; available: {list(gmod.GENERATORS)}")
        if (args.gen_reps or args.gen_motifs) and not hasattr(gmod, "GENOMIC_REPS"):
            ap.error("this module does not handle --gen-reps/--gen-motifs (use corpora_2.py)")
        fe = getattr(gmod, "source_entropy_bits", None)
        if fe is not None:
            floor_bits = fe(args.gen_kind)
            log(f"[source] Shannon entropy: {floor_bits:.2f} bits = {floor_bits / 8:.2f} B/record (asymptotic floor)")
        log(f"[corpus] name = {corpus_name(args)}  (module {os.path.basename(args.gen_module)}, kind={args.gen_kind})")
    else:
        if args.synthetic:
            log("[warning] --synthetic = TOY datasets for script testing: do not use for paper (see --gen-module)")
            recs = synth(args.synthetic, (args.ntrain + args.nval + args.ntest) * 2, 12345)
        elif args.corpus:
            recs = load_records(args.corpus, args.format, args.fasta_headers, args.crop, args)
        else:
            ap.error("--corpus, --synthetic or --gen-module required")
        log(f"[corpus] name = {corpus_name(args)}")
        log(f"[data] {len(recs)} records, {sum(map(len, recs))} bytes, avg {sum(map(len, recs)) / len(recs):.0f} B/rec.")

    mod = None if args.no_codec else load_module(args.codec, "cs_adapt_codec")
    builder = load_builder(args.dict_builder)
    mdlmod = None
    if args.mdl:
        global REF
        args.mdl_modes = [PRUNE_NAMES[x] if x in PRUNE_NAMES else int(x) for x in args.mdl_prune.split(",")]
        mdlmod = load_module(args.mdl, "mdl_selection")
        REF = mdl_codec_name(args, args.mdl_modes[0])
        if mod is None:
            ap.error("--mdl requires the codec (remove --no-codec)")
    ks = [int(k) for k in args.ks.split(",")]
    drifts = [float(x) for x in args.drift.split(",")]

    runsA, runsS, runsT = [], [], []
    runsB = {k: [] for k in ks}
    runsC = {p: [] for p in drifts}
    leaks = []
    for si in range(args.seeds):
        seed = args.seed0 + si
        if gmod is not None:
            train, val, test, leak = gen_split(gmod, args.gen_kind, args.ntrain, args.nval, args.ntest, seed,
                                               not args.keep_leaked)
            if si == 0:
                log(f"[data] {args.gen_kind} seed {seed}: test {len(test)} records, {sum(map(len, test))} B "
                    f"(avg {sum(map(len, test)) / max(1, len(test)):.0f} B/rec.)")
        else:
            train, val, test, leak = make_split(recs, args.ntrain, args.nval, args.ntest, seed, args.group_regex,
                                                not args.keep_leaked)
        leaks.append(leak)
        if args.validate_estimator:
            validate_estimator(train, test, args.budgets[min(1, len(args.budgets) - 1)], seed)
            return
        log(f"\n=== seed {seed}: train={len(train)} val={len(val)} test={len(test)} (exact test∩train leakage before filtering: {100 * leak:.1f}%)")
        if "A" in scen or "S" in scen or "C" in scen or "T" in scen:
            sel = fit_and_select(train, val, len(test), seed, mod, builder, args, mdlmod)
            if "A" in scen:
                runsA.append({m.family: evaluate(m, test) for m in sel})
                log("  [A] evaluated")
            if "S" in scen:
                runsS.append({r["family"]: r for r in run_solid(sel, test, args.extra_cli)})
                log("  [S] evaluated")
            if "C" in scen:
                alpha = bytes(sorted(set(b"".join(train))))
                for p in drifts:
                    rng = random.Random(seed * 1000 + int(p * 1e4))
                    tm = [mutate(x, p, rng, alpha) for x in test]
                    runsC[p].append({m.family: evaluate(m, tm) for m in sel})
                log("  [C] evaluated")
            if "T" in scen and mod is not None:
                cod = next((m for m in sel if m.family == REF), None)
                S_words = cod.extra["S_words"] if cod else set()
                runsT.append(theory_checks(mod, S_words, test, seed=seed))
                log("  [T] evaluated")
        if "B" in scen:
            for k in ks:
                gv, gt = group_concat(val, k), group_concat(test, k)
                if not gv or not gt:
                    continue
                selk = fit_and_select(train, gv, len(gt), seed, mod, builder, args, mdlmod)
                runsB[k].append({m.family: evaluate(m, gt) for m in selk})
                log(f"  [B] k={k} evaluated")

    cname = corpus_name(args)
    base = os.path.join(args.out, f"results_{cname}")
    md, tex, dump = [], [], {}
    md.append(f"# Benchmark Results — {cname}\n\n- seeds: {args.seeds}, train={args.ntrain}, val={args.nval}, test≤{args.ntest}, "
              f"budgets={args.budgets}\n- average exact test∩train leakage before filtering: {100 * statistics.mean(leaks):.1f}%\n"
              + (f"- source: {args.gen_module and os.path.basename(args.gen_module)} / {args.gen_kind}"
                 f"{' / reps=' + args.gen_reps if args.gen_reps else ''}; Shannon floor = {floor_bits / 8:.2f} B/record "
                 f"(≈ {floor_bits / 8 * len(test):.0f} B for {len(test)} records, excluding dictionary)\n" if floor_bits else "") +
              f"- total = payload + dictionary/header + index(lengths); dictionary/header charged for ALL methods at their "
              f"{'stored (zlib-compressed) size' if DICT_CHARGE == 'zlib' else 'RAW (uncompressed) size; codec = whole uncompressed header (alphabet + S + tables)'}"
              f" — both accountings are reported in the tables.\n"
              f"- *(est.)* rows: ideal undecoded code length "
              f"(ctxmodel: +1 B/rec. termination; `~primed`: optimistic bounds, see dedicated table).\n")
    if runsA:
        rows = aggregate(runsA)
        md.append(md_table(rows, "A — per-record, dictionaries learned on train, hyperparameters selected on val"))
        n_test = int(round(statistics.mean(len(next(iter(r.values()))["sizes"]) for r in runsA)))
        avg_len = statistics.mean(next(iter(r.values()))["raw"] / next(iter(r.values()))["n"] for r in runsA)
        meta = dict(corpus=cname, charge=DICT_CHARGE, n_test=n_test, avg_len=f"{avg_len:.1f}", seeds=len(runsA),
                    ntrain=args.ntrain, codec=os.path.basename(args.codec),
                    floor_bits=("%.3f" % floor_bits) if floor_bits else "", gen_reps=args.gen_reps or "",
                    gen_motifs=args.gen_motifs or "")
        tex.append(tex_table(rows, "Scenario A", meta))
        dump["A"] = rows
        if args.nsweep:
            ns = sorted({int(x) for x in args.nsweep_n.split(",")} | {n_test})
            nrows = nsweep(runsA, ns, floor_bytes=(floor_bits / 8 if floor_bits else None))
            md.append(nsweep_md(nrows, ns, n_test, "A — record count sweep n (amortization)"))
            nsweep_files(nrows, ns, base, REF, n_test=n_test, label=args.title)
            dump["A_nsweep"] = nrows
    for k, rr in runsB.items():
        if rr:
            rows = aggregate(rr)
            md.append(md_table(rows, f"B — records = {k} concatenated records (regime transition)"))
            dump[f"B_k{k}"] = rows
    for p, rr in runsC.items():
        if rr and len(drifts) > 0 and "C" in scen:
            rows = aggregate(rr)
            md.append(md_table(rows, f"C — drift: random substitution p={p} (fixed configs, selected at p=0)"))
            dump[f"C_p{p}"] = rows
    if runsS:
        rows = aggregate(runsS, ref="(in block) " + REF)
        md.append(md_table(rows, "S — solid archive (no random access): lrzip, xz, zstd --long, ppmd...", "(in block) " + REF))
        dump["S"] = rows
    if runsT:
        keys = sorted({k for r in runsT for k in r})
        t = "### T — empirical verification of theory (sum over seeds; *_violations must be 0)\n\n"
        for k in keys:
            vals = [r[k] for r in runsT if k in r]
            t += f"- `{k}` = {sum(vals) if k.endswith(('violations', 'cases', 'records')) else statistics.mean(vals):.3f}\n"
        md.append(t)
        dump["T"] = runsT

    text = "\n".join(md)
    print(text)
    open(base + ".md", "w").write(text)
    open(base + ".tex", "w").write("\n".join(tex))
    json.dump(dump, open(base + ".json", "w"), indent=1, default=str)
    log(f"\n→ {base}.md|json|tex" + (f"  +  {base}_nsweep.csv|png" if args.nsweep and runsA else ""))


if __name__ == "__main__":
    main()