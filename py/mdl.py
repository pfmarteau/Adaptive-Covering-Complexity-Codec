#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDL selection of dictionary S with a cost model faithful to the cs_adapt_codec_2.py codec.

v9 (faster): same semantics and results as mdl_corrige_v7_h4.py, algorithms revised
for large corpora / long sequences. Main differences:

  1. Dictionary automaton built ONLY ONCE (on seed S0); removing a word only
     decrements occurrence counters in the suffix link tree (O(|w|^2)),
     instead of rebuilding a complete automaton at each trial (O(M)).
  2. No more per-document prefix automaton: the longest repetition in the prefix is
     found using str.find (C) + exponential/binary search, and only at block
     boundaries; a fallback via prefix automaton remains for very long documents.
  3. Three-level incremental evaluation for each removal trial:
       - document unchanged (credited words still active, identical MTF ranks) -> O(#credited words)
       - parse preserved, only credited word / MTF rank changes                 -> no re-parse
       - a factor D has disappeared                                             -> full re-parse
     "Exact" sweeps (all documents) thus become almost free.
  4. word_freq / factor_docs / dictionary cost updated incrementally (no global
     recounting at each commit); affected documents found by enumerating substrings of the removed word.
  5. Level-based candidate extraction (a priori): a window not repeated at
     length L is not at L+1, hence immediate pruning; no polynomial hashing.
  6. First-occurrence of a factor in the dictionary: cached container word lists
     (static) instead of a linear scan of all words.
  7. Parallelized cross-validation (n_jobs) over (split, lambda).

v10: merger with heuristic "lazy greedy" pruning; common entry point mdl_prune(..., mode=...)
and `prune_mode` parameter in optimize_dictionary_mdl / _cv / select_lambda_cv:

    mode -1: onebyone exact (mdl_backward, v9 behavior unchanged)            -> reference, slowest
    mode  0: removal of never-credited words only (word_freq == 0)            -> almost free
    mode  1: lazy greedy (CELF): phase 0 + one exact scoring + lazy heap     -> ~3x faster, ~+0.2% L
    mode  2: like 1, with relative gain threshold and evaluation budget      -> slightly faster, less precise

The bottleneck of a complete build is generally the SELECTION of the stopping point by the real
codec cost (select_checkpoint_by_codec_cost: one context build per checkpoint, ~25), not the pruning.
`select_search="refine"` option: coarse then fine search (≈ n_checkpoints/2 fewer evaluations if J(k)
is roughly unimodal); `info` receives t_seed / t_prune / t_select.

v11: corrections from paper results audit (RAW header size for all):

  * select_checkpoint_by_codec_cost: J(S) now uses the RAW codec header size
    (alphabet + S + tables, before zlib) -> `header_charge="raw"` (default); "stored" = old behavior
    (zlib header), which underestimates word cost (ratio 1.8 to 7.7) and chooses dictionaries that are too large
    as soon as evaluation charges the raw size.
  * mdl_backward_lazy(extend=True): after the heuristic stopping point (no removal reduces L), removal
    continues (least useful word first) down to `extend_min` words, so that the path contains the true
    minimum of the real codec cost; checkpoint selection decides. Otherwise, the path stops
    before the minimum (synthetic JSON case: minimum of real J at end of path).
  * `lam` / `n_total` unchanged.

Common assumption with the original: the character "\\x01" (word separator) does not appear in the data.
"""

import heapq
import math
import zlib
import os
import random
from bisect import bisect_left, bisect_right
from array import array

try:
    from tqdm import tqdm as _tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False


def _progress(iterable, desc, total=None, enabled=True):
    if enabled and _HAVE_TQDM:
        return _tqdm(iterable, desc=desc, total=total, leave=False)
    return iterable


# Beyond this length, the prefix is followed by a suffix automaton (O(n)) rather than
# str.find (O(n) each, but at C speed): threshold measured empirically.
_FIND_LIMIT = 6000
_SEP = "\x01"

# --------------------------------------------------------------------------------------
# Suffix Automaton (used for dictionary index and fallback on long documents)
# --------------------------------------------------------------------------------------


class SuffixAutomaton:
    __slots__ = ("trans", "link", "length", "last")

    def __init__(self):
        self.trans = [{}]
        self.link = [-1]
        self.length = [0]
        self.last = 0

    def extend(self, c):
        trans, link, length = self.trans, self.link, self.length
        cur = len(trans)
        trans.append({})
        length.append(length[self.last] + 1)
        link.append(-1)
        p = self.last
        while p != -1 and c not in trans[p]:
            trans[p][c] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = trans[p][c]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                clone = len(trans)
                trans.append(dict(trans[q]))
                length.append(length[p] + 1)
                link.append(link[q])
                while p != -1 and trans[p].get(c) == q:
                    trans[p][c] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone
        self.last = cur

    def longest_match(self, s, start):
        state, i, n = 0, start, len(s)
        trans = self.trans
        while i < n:
            nxt = trans[state].get(s[i])
            if nxt is None:
                break
            state = nxt
            i += 1
        return i - start


def build_dict_automaton(pruned_words):
    sam = SuffixAutomaton()
    for w in sorted(pruned_words):
        for c in w:
            sam.extend(c)
        sam.extend(_SEP)
    return sam


# --------------------------------------------------------------------------------------
# Dictionary index with activatable / deactivatable words (built only once)
# --------------------------------------------------------------------------------------


class DictIndex:
    """Suffix automaton of seed S0 + "active" occurrence counters.

    cnt[v] = number of end positions (in ACTIVE words) of state v. A substring is a
    factor of the current dictionary  <=>  the state containing it has cnt > 0. Disabling a word w
    decrements cnt along the suffix links of each of its prefix states (O(|w|^2) worst case).
    """

    def __init__(self, words):
        ws = sorted(set(words), key=lambda w: (len(w), w))       # order (|w|, w) from the paper
        self.words = ws
        self.max_len = max((len(w) for w in ws), default=0)
        sam = SuffixAutomaton()
        pref = {}
        for w in ws:
            states = []
            for c in w:
                sam.extend(c)
                states.append(sam.last)
            pref[w] = states
            sam.extend(_SEP)
        self.trans, self.link, length = sam.trans, sam.link, sam.length
        cnt = [0] * len(self.trans)
        for states in pref.values():
            for st in states:
                cnt[st] += 1
        for v in sorted(range(1, len(cnt)), key=length.__getitem__, reverse=True):
            cnt[self.link[v]] += cnt[v]
        self.cnt, self.pref = cnt, pref
        self.active = set(ws)
        self.sorted_active = sorted(ws)
        self._rank_map = None            # word -> rank among active words (rebuilt on demand)
        # concatenated string to quickly find words containing a factor
        self._big = _SEP.join(ws)
        self._starts, off = [], 0
        for w in ws:
            self._starts.append(off)
            off += len(w) + 1
        self._contain = {}

    # ---- activation -------------------------------------------------------------------
    def deactivate(self, words):
        cnt, link, active, sa = self.cnt, self.link, self.active, self.sorted_active
        for w in words:
            if w not in active:
                continue
            active.discard(w)
            self._rank_map = None
            del sa[bisect_left(sa, w)]
            for st in self.pref[w]:
                x = st
                while x > 0:
                    cnt[x] -= 1
                    x = link[x]

    def activate(self, words):
        cnt, link, active, sa = self.cnt, self.link, self.active, self.sorted_active
        for w in words:
            if w in active or w not in self.pref:
                continue
            active.add(w)
            self._rank_map = None
            sa.insert(bisect_left(sa, w), w)
            for st in self.pref[w]:
                x = st
                while x > 0:
                    cnt[x] += 1
                    x = link[x]

    # ---- queries ---------------------------------------------------------------------
    def match(self, s, start):
        """Length of the longest factor of an active word starting at s[start]."""
        trans, cnt = self.trans, self.cnt
        st, i, n = 0, start, len(s)
        while i < n:
            nx = trans[st].get(s[i])
            if nx is None or cnt[nx] <= 0:
                break
            st = nx
            i += 1
        return i - start

    longest_match = match            # compatibility with old API (SuffixAutomaton)

    def first_match(self, factor):
        """First ACTIVE word, in order (|w|, w), containing `factor` (None otherwise)."""
        lst = self._contain.get(factor)
        if lst is None:
            big, starts, ws = self._big, self._starts, self.words
            idx, pos = set(), big.find(factor)
            while pos != -1:
                idx.add(bisect_right(starts, pos) - 1)
                pos = big.find(factor, pos + 1)
            lst = [ws[k] for k in sorted(idx)]
            self._contain[factor] = lst
        active = self.active
        for w in lst:
            if w in active:
                return w
        return None

    def rank(self, w):
        return bisect_left(self.sorted_active, w)

    def rank_map(self):
        rm = self._rank_map
        if rm is None:
            rm = self._rank_map = {w: j for j, w in enumerate(self.sorted_active)}
        return rm


class FastFactorSet:
    """Compatibility with old API: wrapper for a fully active DictIndex."""
    __slots__ = ("ix", "S")

    def __init__(self, S):
        self.S = set(S)
        self.ix = DictIndex(self.S)

    @property
    def dict_aut(self):
        return self.ix if self.S else None

    def first_match(self, factor):
        return self.ix.first_match(factor)


# --------------------------------------------------------------------------------------
# Greedy segmentation of a document (D / S / L) and recording of its parse
# --------------------------------------------------------------------------------------

K_L, K_D, K_S = 1, 2, 3


def _mk(s, i, ml, dl, kind):
    if kind == K_D:
        return (i, ml, dl, K_D, s[i:i + ml])
    if kind == K_S:
        occ = s.rfind(s[i:i + ml], 0, i)
        return (i, ml, dl, K_S, (i - occ) if occ >= 0 else 1)
    return (i, 1, dl, K_L, s[i])


def _block_at(s, i, n, dl, find):
    """Block chosen at i (str.find mode). Original rule:
        D if dl >= max(pl, 1); else S if pl > 1; else L   (pl = longest repetition of the prefix)."""
    rem = n - i
    pl = dl                              # if pl <= dl we only need the comparison
    if dl < rem:
        t = dl + 1                       # does a repetition of length t exist in s[:i]?
        if find(s[i:i + t], 0, i) >= 0:
            lo, step = t, 1              # exponential search then binary search for maximum
            while lo < rem:
                cand = min(lo + step, rem)
                if find(s[i:i + cand], 0, i) >= 0:
                    lo, step = cand, step * 2
                else:
                    hi = cand - 1
                    while lo < hi:
                        mid = (lo + hi + 1) >> 1
                        if find(s[i:i + mid], 0, i) >= 0:
                            lo = mid
                        else:
                            hi = mid - 1
                    break
            pl = lo
    if dl >= 1 and dl >= pl:
        return (i, dl, dl, K_D, s[i:i + dl])
    if pl > 1:
        occ = s.rfind(s[i:i + pl], 0, i)
        return (i, pl, dl, K_S, (i - occ) if occ >= 0 else 1)
    return (i, 1, dl, K_L, s[i])


def _segment(s, ix, start=0):
    """List of blocks (i, ml, dl, kind, x). dl = longest dictionary factor at i (even when
    an S block is chosen); ml = advanced length; x = factor (D) / distance (S) / character (L)."""
    n = len(s)
    out = []
    match = ix.match if ix is not None else None
    if n <= _FIND_LIMIT:
        find, i = s.find, start
        while i < n:
            b = _block_at(s, i, n, match(s, i) if match else 0, find)
            out.append(b)
            i += b[1]
    else:                                # very long document: prefix automaton (O(n))
        psam = SuffixAutomaton()
        for k in range(start):
            psam.extend(s[k])
        i = start
        while i < n:
            dl = match(s, i) if match else 0
            pl = psam.longest_match(s, i)
            if dl >= 1 and dl >= pl:
                kind, ml = K_D, dl
            elif pl > 1:
                kind, ml = K_S, pl
            else:
                kind, ml = K_L, 1
            out.append(_mk(s, i, ml, dl, kind))
            for k in range(i, i + ml):
                psam.extend(s[k])
            i += ml
    return out


def _entropy_cost(counts: dict) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return sum(c * math.log2(total / c) for c in counts.values() if c > 0)


class _Doc:
    """Document parse: static parts (do not depend on dictionary state as long as
    D factors exist) + dynamic parts (credited word, MTF rank, offset)."""
    __slots__ = ("blocks", "uf", "fid", "ucnt", "factors", "bt", "lit", "lenD", "lenS", "dist", "ent_stat",
                 "mtf", "off", "cw", "cr", "uw", "ev_base", "ev_word", "ent_off", "cost")


def _eval_dyn(uf, fid, ucnt, ix, rank_fn, nS, strict, mtf_init=None, known=None):
    """MTF / offset statistics and credited words of a document.

    uf   : distinct D factors, in order of first appearance;
    fid  : for each D block, the index of its factor in uf; ucnt: multiplicity of each factor.
    The credited word, its rank, and the offset are calculated once per distinct factor; the loop over
    blocks handles integers only.

    "Compact" MTF: only already used words are tracked (`front` list, most recent to
    oldest); a never-used word of rank wi is at position len(front) + wi - #(used < wi),
    exactly like in the original list(range(nS)).

    Events: for each D block, (ev_base, ev_word). Reuse: index = ev_base (ev_word = -1).
    First use of word cw[j]: index = ev_base + rank(cw[j]). The term ev_base depends only on
    the relative order of words, so it remains valid when removals merely shift ranks.
    """
    U = len(uf)
    Uwi, Uo, Uj, uw = [-1] * U, [0] * U, [0] * U, [None] * U
    cw, cr, rk = [], [], {}
    first_match = ix.first_match
    kget = known.get if known else None       # factor -> already known word (and still active)
    for k in range(U):
        f = uf[k]
        w = kget(f) if kget else None
        if w is None:
            w = first_match(f)
        if w is None:
            if strict:
                return None
            continue
        uw[k] = w
        t = rk.get(w)
        if t is None:
            wi = rank_fn(w)
            t = rk[w] = (wi, len(cw))
            cw.append(w)
            cr.append(wi)
        Uwi[k], Uj[k], Uo[k] = t[0], t[1], w.find(f)
    off_st = {}                      # same insertion order as original (first appearance)
    for k in range(U):
        if Uwi[k] >= 0:
            o = Uo[k]
            off_st[o] = off_st.get(o, 0) + ucnt[k]
    mtf_st, ev_base, ev_word = {}, [], []
    if mtf_init is not None:         # generic path (imposed initial MTF state), complete list
        mtf = list(mtf_init)
        for k in fid:
            wi = Uwi[k]
            if wi < 0:
                continue
            try:
                idx = mtf.index(wi)
                mtf.insert(0, mtf.pop(idx))
            except ValueError:
                idx = 0
            mtf_st[idx] = mtf_st.get(idx, 0) + 1
        return mtf_st, off_st, tuple(cw), tuple(cr), uw, ev_base, ev_word
    front, used = [], []
    for k in fid:
        wi = Uwi[k]
        if wi < 0:
            continue
        if front and front[0] == wi:                  # same word as previous D block
            idx = 0
            ev_base.append(0)
            ev_word.append(-1)
        else:
            try:
                idx = front.index(wi)
                del front[idx]
                front.insert(0, wi)
                ev_base.append(idx)
                ev_word.append(-1)
            except ValueError:
                if wi >= nS:
                    idx = 0
                    ev_base.append(0)
                    ev_word.append(-1)
                else:
                    pos = bisect_left(used, wi)
                    idx = len(front) + wi - pos
                    used.insert(pos, wi)
                    front.insert(0, wi)
                    ev_base.append(len(front) - 1 - pos)
                    ev_word.append(Uj[k])
        mtf_st[idx] = mtf_st.get(idx, 0) + 1
    return mtf_st, off_st, tuple(cw), tuple(cr), uw, ev_base, ev_word


def _finish(doc, dyn, e_off=None):
    doc.mtf, doc.off, doc.cw, doc.cr, doc.uw, doc.ev_base, doc.ev_word = dyn
    doc.ent_off = _entropy_cost(doc.off) if e_off is None else e_off
    e_bt, e_lit, e_lenD, e_lenS, e_dist = doc.ent_stat
    # same summation order as original: bt, lit, mtf, len_D, len_S, off, dist
    tot = 0 + e_bt
    tot += e_lit
    tot += _entropy_cost(doc.mtf)
    tot += e_lenD
    tot += e_lenS
    tot += doc.ent_off
    tot += e_dist
    doc.cost = tot / 8.0 + 3
    return doc


def _build_doc(s, blocks, ix, rank_fn=None, nS=None, known=None):
    """Document statistics from its segmentation (single pass over blocks)."""
    bt, lit, lenD, lenS, dist = {}, {}, {}, {}, {}
    pos, uf, fid, ucnt = {}, [], [], []
    prev = 0
    for (i, ml, dl, kind, x) in blocks:
        if kind == K_D:
            k = pos.get(x)
            if k is None:
                k = len(uf)
                pos[x] = k
                uf.append(x)
                ucnt.append(0)
            fid.append(k)
            ucnt[k] += 1
            lenD[ml] = lenD.get(ml, 0) + 1
        elif kind == K_S:
            lenS[ml] = lenS.get(ml, 0) + 1
            dist[x] = dist.get(x, 0) + 1
        else:
            lit[x] = lit.get(x, 0) + 1
        key = prev * 4 + kind
        bt[key] = bt.get(key, 0) + 1
        prev = kind
    doc = _Doc()
    doc.blocks = blocks
    doc.uf, doc.fid, doc.ucnt, doc.factors = uf, fid, ucnt, set(uf)
    doc.bt, doc.lit, doc.lenD, doc.lenS, doc.dist = bt, lit, lenD, lenS, dist
    doc.ent_stat = (_entropy_cost(bt), _entropy_cost(lit), _entropy_cost(lenD),
                    _entropy_cost(lenS), _entropy_cost(dist))
    if rank_fn is None:
        rank_fn, nS = ix.rank_map().__getitem__, len(ix.sorted_active)
    return _finish(doc, _eval_dyn(uf, fid, ucnt, ix, rank_fn, nS, strict=False, known=known))


def _parse_doc(s, ix, rank_fn=None, nS=None, blocks=None, known=None):
    """Document parse under the current state of `ix` (`blocks`: precomputed segmentation)."""
    if blocks is None:
        blocks = _segment(s, ix)
    return _build_doc(s, blocks, ix, rank_fn, nS, known)


def _doc_stats_dict(doc):
    return {"bt": doc.bt, "lit": doc.lit, "mtf": doc.mtf, "len_D": doc.lenD,
            "len_S": doc.lenS, "off": doc.off, "dist": doc.dist}


def parse_and_collect_stats(s, factor_set, S_list, mtf_state=None, track_used=None):
    """Preserved API: document statistics (keys of 'bt' = integers prev*4+type)."""
    ix = factor_set.ix if factor_set is not None else DictIndex(())
    rank_of = {w: j for j, w in enumerate(S_list)}
    rank_fn, nS = (lambda w: rank_of.get(w, 0)), len(S_list)
    doc = _parse_doc(s, ix, rank_fn, nS)
    if mtf_state is not None:
        dyn = _eval_dyn(doc.uf, doc.fid, doc.ucnt, ix, rank_fn, nS, False, mtf_state)
        _finish(doc, dyn)
    if track_used is not None:
        track_used.update(doc.cw)
    return _doc_stats_dict(doc)


def data_cost_codec(corpus, factor_set, S_list, track_used=None):
    if not corpus:
        return 0.0
    ix = factor_set.ix
    rank_of = {w: j for j, w in enumerate(S_list)}
    rank_fn, nS = (lambda w: rank_of.get(w, 0)), len(S_list)
    merged = {"bt": {}, "lit": {}, "mtf": {}, "len_D": {}, "len_S": {}, "off": {}, "dist": {}}
    for s in corpus:
        doc = _parse_doc(s, ix, rank_fn, nS)
        st = _doc_stats_dict(doc)
        for k, dst in merged.items():
            for sym, count in st[k].items():
                dst[sym] = dst.get(sym, 0) + count
        if track_used is not None:
            track_used.update(doc.cw)
    total_bits = sum(_entropy_cost(merged[k]) for k in merged)
    return (total_bits / 8.0) + (len(corpus) * 3)


def dict_cost_codec(S):
    return sum(1 + len(w.encode("utf-8")) for w in S) + 8.0


# --------------------------------------------------------------------------------------
# Backward pruning: three-level incremental evaluation
# --------------------------------------------------------------------------------------


class _PruneState:
    def __init__(self, S, train, lam, counter):
        self.train, self.lam, self.counter = train, lam, counter
        self.ix = DictIndex(S)
        self.dict_bytes = sum(1 + len(w.encode("utf-8")) for w in self.ix.words)
        self.docs = [None] * len(train)
        self.full_scan()

    @property
    def S(self):
        return self.ix.active

    # ---- evaluation of a document under the current index state ------------------------
    def _eval(self, d):
        doc, ix = self.docs[d], self.ix
        if doc is None:
            self.counter[1] += 1
            return _parse_doc(self.train[d], ix)
        sa, rm = ix.sorted_active, ix.rank_map()
        all_active, same = True, True
        for w, r in zip(doc.cw, doc.cr):
            x = rm.get(w)
            if x is None:                             # credited word removed
                all_active = False
                break
            if x != r:
                same = False
        if all_active:
            if same:                                  # level 1: nothing changes
                self.counter[3] += 1
                return doc
            # level 2: same credited words, same offsets; only initial ranks of words at their
            # first use shift -> reconstruct MTF counter, without re-parse.
            self.counter[2] += 1
            cr = tuple([rm[w] for w in doc.cw])
            mtf_st = {}
            for b, j in zip(doc.ev_base, doc.ev_word):
                k = b if j < 0 else b + cr[j]
                mtf_st[k] = mtf_st.get(k, 0) + 1
            new = _Doc()
            for a in ("blocks", "uf", "fid", "ucnt", "factors", "bt", "lit", "lenD", "lenS", "dist",
                      "ent_stat", "cw", "uw", "ev_base", "ev_word"):
                setattr(new, a, getattr(doc, a))
            return _finish(new, (mtf_st, doc.off, doc.cw, cr, doc.uw, doc.ev_base, doc.ev_word), doc.ent_off)
        # A credited word has disappeared: recalculate credited word only for concerned factors.
        known, lost, fm = {}, set(), ix.first_match
        for f, w in zip(doc.uf, doc.uw):
            if w is not None and w in rm:
                known[f] = w
            else:
                nw = fm(f)
                if nw is None:
                    lost.add(f)
                else:
                    known[f] = nw
        if lost:                                      # level 3: a D factor has disappeared
            self.counter[1] += 1
            return self._reparse(d, doc, lost, known)
        self.counter[2] += 1                          # level 2b: same parse, other credited word(s)
        dyn = _eval_dyn(doc.uf, doc.fid, doc.ucnt, ix, rm.__getitem__, len(sa), True, known=known)
        new = _Doc()
        for a in ("blocks", "uf", "fid", "ucnt", "factors", "bt", "lit", "lenD", "lenS", "dist", "ent_stat"):
            setattr(new, a, getattr(doc, a))
        return _finish(new, dyn)

    def _reparse(self, d, doc, lost, known):
        """Removing words can only change the parse starting from the first D block whose factor has
        disappeared. We re-segment from that block only until falling back onto an old parse boundary
        (segmentation depends only on position), then copy old blocks until the next lost factor, and so on."""
        s, ix = self.train[d], self.ix
        n, old = len(s), doc.blocks
        lost_idx = [k for k, b in enumerate(old) if b[3] == K_D and b[4] in lost]
        if n > _FIND_LIMIT or not lost_idx:
            return _parse_doc(s, ix, known=known)
        starts = [b[0] for b in old]
        nst, nl = len(starts), len(lost_idx)
        find, match = s.find, ix.match
        out = old[:lost_idx[0]]
        li, i = 0, old[lost_idx[0]][0]
        while i < n:
            b = _block_at(s, i, n, match(s, i), find)
            out.append(b)
            i += b[1]
            if i >= n:
                break
            k = bisect_left(starts, i)
            if k < nst and starts[k] == i:            # fell back onto an old parse boundary
                while li < nl and lost_idx[li] < k:
                    li += 1
                if li < nl:
                    j = lost_idx[li]
                    out.extend(old[k:j])
                    i = old[j][0]
                else:
                    out.extend(old[k:])
                    i = n
        return _parse_doc(s, ix, blocks=out, known=known)

    def _build_index(self):
        self.factor_docs = {}
        self.word_freq = {w: 0 for w in self.ix.active}
        for d, doc in enumerate(self.docs):
            for x in doc.factors:
                self.factor_docs.setdefault(x, set()).add(d)
            for w in doc.cw:
                if w in self.word_freq:
                    self.word_freq[w] += 1

    def full_scan(self):
        self.counter[0] += len(self.train)
        for d in range(len(self.train)):
            self.docs[d] = self._eval(d)
        self.cost = [doc.cost for doc in self.docs]
        self.data = sum(self.cost)
        self._build_index()

    def L(self):
        return (self.dict_bytes + 8.0) + self.lam * self.data

    def _affected(self, words):
        aff, fd = set(), self.factor_docs
        for w in words:
            lw = len(w)
            for a in range(lw):
                for b in range(a + 1, lw + 1):
                    docs = fd.get(w[a:b])
                    if docs:
                        aff |= docs
        return aff

    def try_remove(self, words, incremental=True):
        """Deactivates `words` and evaluates. Caller MUST then call commit() or rollback()."""
        words = [w for w in set(words) if w in self.ix.active]
        aff = sorted(self._affected(words)) if incremental else range(len(self.train))
        self.counter[0] += len(aff)
        self.ix.deactivate(words)
        changed = {}
        for d in aff:
            new = self._eval(d)
            if new is not self.docs[d]:
                changed[d] = new
        data_c = self.data
        for d, new in changed.items():
            data_c = data_c - self.cost[d] + new.cost
        dbytes = self.dict_bytes - sum(1 + len(w.encode("utf-8")) for w in words)
        return (dbytes + 8.0) + self.lam * data_c, (words, changed, data_c, dbytes)

    def rollback(self, payload):
        self.ix.activate(payload[0])

    def commit(self, payload):
        words, changed, data_c, dbytes = payload
        fd, wf = self.factor_docs, self.word_freq
        for d, new in changed.items():
            old = self.docs[d]
            if new.factors is not old.factors:
                for x in old.factors:
                    docs = fd.get(x)
                    if docs is not None:
                        docs.discard(d)
                        if not docs:
                            del fd[x]
                for x in new.factors:
                    fd.setdefault(x, set()).add(d)
            for w in old.cw:
                if w in wf:
                    wf[w] -= 1
            for w in new.cw:
                if w in wf:
                    wf[w] += 1
            self.docs[d] = new
            self.cost[d] = new.cost
        for w in words:
            wf.pop(w, None)
        self.data, self.dict_bytes = data_c, dbytes


def mdl_backward(S_init, train, lam, return_trace=False, progress=True, batch_pct=0.15,
                 strategy="onebyone", order="score", incremental=True, stats=None, max_sweeps=100):
    """Backward pruning of S under the codec-aware MDL objective (same parameters as v7).

    strategy: "batch" | "onebyone" (default) | "hybrid"; order: "score" | "len".
    stats   : optional dict filled with n_parses (considered documents), n_full_parses,
              n_stats_only (parse kept), n_unchanged, L_final, sweeps.
    """
    counter = [0, 0, 0, 0]     # considered, full parses, MTF-only re-evaluations, unchanged
    st = _PruneState(S_init, train, lam, counter)
    cur = st.L()
    trace = [(len(st.S), cur)]
    if stats is not None and 'snapshots' in stats:
        stats['snapshots'].append(frozenset(st.S))

    def ranked(kind):
        if kind == "score":
            return sorted(st.S, key=lambda w: (st.word_freq.get(w, 0) * (len(w) ** 1.5), len(w), w))
        return sorted(st.S, key=lambda w: (len(w), w))

    def accept(c, payload):
        nonlocal cur
        st.commit(payload)
        cur = c
        trace.append((len(st.S), cur))
        if stats is not None and 'snapshots' in stats:
            stats['snapshots'].append(frozenset(st.S))

    if strategy in ("batch", "hybrid"):
        improved, pass_num = True, 1
        while improved and len(st.S) > 1:
            improved = False
            cands = ranked("score")
            k = max(1, int(len(cands) * batch_pct))
            while k >= 1:
                c, payload = st.try_remove(set(cands[:k]), incremental)
                if c <= cur:
                    accept(c, payload)
                    improved = True
                    if progress and _HAVE_TQDM:
                        print(f"Pass {pass_num}: -{k} words | |S|={len(st.S)} | Cost={cur:.2f}")
                    break
                st.rollback(payload)
                k //= 2
            pass_num += 1

    sweeps = 0
    if strategy in ("onebyone", "hybrid"):
        # _PruneState.__init__ already performs a full_scan(). Doing it here
        # was redundant for onebyone; hybrid only needs to resynchronize
        # after its batch phase.
        if strategy == "hybrid" and trace[-1][0] != len(st.S):
            st.full_scan()
        cur = st.L()

        def sweep(inc):
            nonlocal cur
            removed = False
            for w in ranked(order):
                if w not in st.S or len(st.S) <= 1:
                    continue
                c, payload = st.try_remove({w}, inc)
                if c <= cur:
                    accept(c, payload)
                    removed = True
                else:
                    st.rollback(payload)
            st.full_scan()                 # exact resynchronization
            cur = st.L()
            return removed

        while sweeps < max_sweeps and len(st.S) > 1:
            improved = True
            while improved and sweeps < max_sweeps and len(st.S) > 1:
                sweeps += 1
                improved = sweep(True)
            sweeps += 1
            if not sweep(False):
                break

    if stats is not None:
        # The last onebyone sweep is already synchronized. Replaying full_scan()
        # does not change the result, only diagnostic counters.
        if strategy == "onebyone":
            counter[0] += len(train)
            counter[3] += len(train)
        stats.update(n_parses=counter[0], n_full_parses=counter[1], n_stats_only=counter[2],
                     n_unchanged=counter[3], L_final=st.L(), sweeps=sweeps)
    S = set(st.S)
    return (S, trace) if return_trace else S


# --------------------------------------------------------------------------------------
# "Lazy greedy" heuristic pruning (CELF) and common entry point mdl_prune
# --------------------------------------------------------------------------------------


def snapshots_from_log(S0, removed, n_checkpoints=25):
    """Reconstructs ~n_checkpoints dictionaries of the pruning path from the removal log
    (instead of one frozenset per accepted removal, quadratic)."""
    K = len(removed)
    idx = sorted({round(i * K / max(1, n_checkpoints - 1)) for i in range(n_checkpoints)})
    cur, snaps, k = set(S0), [], 0
    for t in idx:
        while k < t:
            cur.discard(removed[k])
            k += 1
        snaps.append(frozenset(cur))
    return snaps


class _LazySnapshots:
    """Lazy sequence of dictionaries of the pruning path: snapshot k = S0 without the first k removed words.
    O(|S0|) per access, no stored copy (instead of one frozenset per removal)."""
    def __init__(self, S0, removed):
        self.S0, self.removed = frozenset(S0), list(removed)

    def __len__(self):
        return len(self.removed) + 1

    def __getitem__(self, k):
        if k < 0:
            k += len(self)
        if not 0 <= k < len(self):
            raise IndexError(k)
        return frozenset(self.S0.difference(self.removed[:k]))


def mdl_backward_lazy(S_init, train, lam, dead_only=False, keep_top=0.0, min_gain=0.0, min_gain_rel=0.0,
                      max_evals=None, evals_per_word=None, stats=None, n_checkpoints=25,
                      extend=False, extend_min=1):
    """Heuristic pruning.

    Phase 0: bulk removal of never-credited words (parse does not change); if dead_only, stop.
    Phase 1: ONE exact scoring delta(w) = L(S\\{w}) - L(S) (with rollback) of each candidate word.
    Phase 2: min heap on delta; only the head is refreshed. Heuristic assumption (not guaranteed):
             removing other words only makes a word more useful -> delta only increases, an outdated value
             is an optimistic bound; refreshed head > 0 => discarded definitively.
    Stop   : head > -gain_min, or evaluation budget exhausted.

    keep_top      : fraction of words with highest freq*len^1.5 score ever tested (discouraged: 0).
    min_gain      : minimal gain (in bytes) to remove a word; min_gain_rel: same, relative to initial L.
    max_evals     : absolute budget of exact evaluations; evals_per_word: budget relative to |S0|.
    Optimality losses: single initial scoring, definitively discarded words, no catch-up exact sweep.
    """
    counter = [0, 0, 0, 0]
    st = _PruneState(S_init, train, lam, counter)
    S0 = set(st.S)
    cur = st.L()
    gain_min = max(min_gain, min_gain_rel * cur)
    if max_evals is None and evals_per_word is not None:
        max_evals = int(evals_per_word * len(S0))
    removed, evals, stop_idx = [], 0, None

    # Phase 0
    dead = {w for w, f in st.word_freq.items() if f == 0}
    if dead and len(dead) < len(st.S):
        c, p = st.try_remove(dead)
        if c <= cur:
            st.commit(p)
            cur = c
            removed.extend(sorted(p[0]))
        else:
            st.rollback(p)

    if not dead_only:
        # Phase 1
        wf = st.word_freq
        order = sorted(st.S, key=lambda w: (wf.get(w, 0) * len(w) ** 1.5, len(w), w))
        cands = order[:max(1, int(len(order) * (1.0 - keep_top)))]
        heap = []
        for w in cands:
            if max_evals is not None and evals >= max_evals:
                break
            c, p = st.try_remove({w})
            st.rollback(p)
            evals += 1
            heap.append((c - cur, w))
        heapq.heapify(heap)

        # Phase 2 (extend=True: do not stop where L stops decreasing; removal continues
        # down to `extend_min` words, and real codec cost (checkpoint selection) chooses)
        while heap and len(st.S) > max(1, extend_min if extend else 1):
            d, w = heapq.heappop(heap)
            if d > -gain_min and not extend:
                break
            if d > -gain_min and stop_idx is None:
                stop_idx = len(removed)
            if w not in st.S:
                continue
            if max_evals is not None and evals >= max_evals:
                break
            c, p = st.try_remove({w})
            evals += 1
            nd = c - cur
            nxt = heap[0][0] if heap else float("inf")
            if (nd <= -gain_min or extend) and nd <= nxt:
                if nd > -gain_min and stop_idx is None:
                    stop_idx = len(removed)
                st.commit(p)
                cur = c
                removed.append(w)
            else:
                st.rollback(p)
                if nd <= -gain_min or extend:
                    heapq.heappush(heap, (nd, w))

    st.full_scan()                                  # single exact resynchronization
    cur = st.L()
    if stats is not None:
        stats.update(n_parses=counter[0], n_full_parses=counter[1], n_stats_only=counter[2],
                     n_unchanged=counter[3], L_final=cur, sweeps=0, evals=evals, n_removed=len(removed))
        stats["L_stop_index"] = stop_idx if extend else len(removed)
        if "snapshots" in stats:
            stats["snapshots"] = _LazySnapshots(S0, removed)
    return set(st.S)


# mode -> mdl_backward_lazy parameters (mode -1 delegates to mdl_backward)
PRUNE_MODES = {
    0: dict(dead_only=True),
    1: dict(),
    2: dict(min_gain_rel=1e-5, evals_per_word=2.5),
    3: dict(extend=True),                         # lazy greedy + extended path (recommended with header_charge="raw")
}


def mdl_prune(S_init, train, lam, mode=-1, progress=True, return_trace=False, stats=None, n_checkpoints=25,
              **kw):
    """Common entry point for dictionary pruning.

    mode -1: exact onebyone (mdl_backward); kw: strategy, order, batch_pct, incremental, max_sweeps
    mode  0: only never-credited words (word_freq == 0) are removed
    mode  1: lazy greedy (CELF)
    mode  2: lazy greedy with relative gain threshold and evaluation budget
    For modes >= 0, kw can override: keep_top, min_gain, min_gain_rel, max_evals, evals_per_word.
    `stats` (dict) is filled in all modes; with 'snapshots' key it receives dictionaries of the
    pruning path (all for mode -1, ~n_checkpoints for others). return_trace: mode -1 only.
    """
    if mode == -1:
        return mdl_backward(S_init, train, lam, return_trace=return_trace, progress=progress, stats=stats, **kw)
    if mode not in PRUNE_MODES:
        raise ValueError(f"unknown pruning mode: {mode!r} (expected: -1, {', '.join(map(str, PRUNE_MODES))})")
    if return_trace:
        raise ValueError("return_trace is only available in mode -1")
    params = {**PRUNE_MODES[mode], **kw}
    return mdl_backward_lazy(S_init, train, lam, stats=stats, n_checkpoints=n_checkpoints, **params)


# --------------------------------------------------------------------------------------
# Candidate extraction & chunks
# --------------------------------------------------------------------------------------

def chunk_corpus(corpus, max_seq_len, overlap=0):
    if max_seq_len is None:
        return list(corpus), list(range(len(corpus)))
    assert overlap < max_seq_len, "overlap must be strictly less than max_seq_len"
    chunks, origins = [], []
    step = max_seq_len - overlap
    for idx, s in enumerate(corpus):
        if len(s) <= max_seq_len:
            chunks.append(s)
            origins.append(idx)
            continue
        i = 0
        while True:
            chunks.append(s[i:i + max_seq_len])
            origins.append(idx)
            if i + max_seq_len >= len(s):
                break
            i += step
    return chunks, origins


def extract_maximal_candidates(corpus, lmin=3, lmax=64, min_docs=2, origins=None, progress=True):
    """Same candidates as v7, computed level by level.

    Phase 1 (increasing L): rep[g] = largest length L <= lmax such that the window of length L
    starting at g appears in >= min_docs distinct origin documents (monotone property:
    a window not repeated at L is not at L+1, so only surviving windows are re-examined).
    Phase 2 (decreasing L): selection with coverage, exactly like v7, iterating only over
    repeated and not yet covered windows (a covered window remains so for any L' < L).
    """
    docs = list(corpus)
    n_docs = len(docs)
    origins = list(origins) if origins is not None else list(range(n_docs))
    lens = [len(s) for s in docs]
    max_len = max(lens, default=0)
    if max_len == 0:
        return set()
    top = min(lmax, max_len)

    T = "".join(docs)
    total = len(T)
    org, dend = array("I"), array("I")
    start, offs = 0, []
    for d, n in enumerate(lens):
        offs.append(start)
        org.extend(array("I", [origins[d]]) * n)
        dend.extend(array("I", [start + n]) * n)
        start += n

    # ---- phase 1: repeated windows, level by level ------------------------------------
    rep = bytearray(total)
    cur = []
    for d, n in enumerate(lens):
        if n >= lmin:
            cur.extend(range(offs[d], offs[d] + n - lmin + 1))
    joiners = [None] * (top + 2)
    prev_keep, prev_L, L = None, 0, lmin
    bar = _tqdm(total=top - lmin + 1, desc="candidate extraction", leave=False) \
        if (progress and _HAVE_TQDM) else None
    while L <= top and cur:
        # Avoid intermediate `contents` list: on large corpora it
        # duplicates millions of Python references at each level.
        # min_docs<=2 path is the default and only requires
        # a single small dictionary `content -> first origin`.
        if min_docs <= 2:
            first, multi = {}, set()
            Tloc, orgloc = T, org
            for g in cur:
                c = Tloc[g:g + L]
                o = orgloc[g]
                f = first.get(c)
                if f is None:
                    first[c] = o
                elif f != o:
                    multi.add(c)
            keep = [g for g in cur if Tloc[g:g + L] in multi]
        else:
            sets = {}
            Tloc, orgloc = T, org
            for g in cur:
                c = Tloc[g:g + L]
                ss = sets.get(c)
                if ss is None:
                    sets[c] = {orgloc[g]}
                else:
                    ss.add(orgloc[g])
            multi = {c for c, ss in sets.items() if len(ss) >= min_docs}
            keep = [g for g in cur if Tloc[g:g + L] in multi]
        for g in keep:
            rep[g] = L
        if prev_keep is not None:                       # windows whose max length is L-1
            joiners[prev_L] = [g for g in prev_keep if rep[g] == prev_L]
        prev_keep, prev_L = keep, L
        cur = [g for g in keep if g + L + 1 <= dend[g]]
        L += 1
        if bar:
            bar.update(1)
    if bar:
        bar.close()
    if prev_keep:
        joiners[prev_L] = [g for g in prev_keep if rep[g] == prev_L]

    # ---- phase 2: selection with coverage (same semantics as v7) -----------------------
    cov = bytearray(total)
    maximal, alive = set(), []
    for L in range(top, lmin - 1, -1):
        if joiners[L]:
            alive.extend(joiners[L])
        if not alive:
            continue
        alive = [g for g in alive if cov[g] < L]
        groups = {}
        Tloc = T
        for g in alive:
            content = Tloc[g:g + L]
            gs = groups.get(content)
            if gs is None:
                groups[content] = [g]
            else:
                gs.append(g)
        for content, gs in groups.items():
            if len({org[g] for g in gs}) < min_docs:
                continue
            maximal.add(content)
            for g in gs:
                for o in range(L):
                    if cov[g + o] < L - o:
                        cov[g + o] = L - o
        del groups

    # ---- subsumption pruning ------------------------------------------------------------
    survivors = []
    surv_sam = SuffixAutomaton()
    for w in sorted(maximal, key=lambda w: (-len(w), w)):
        if surv_sam.longest_match(w, 0) == len(w):
            continue
        survivors.append(w)
        for c in w:
            surv_sam.extend(c)
        surv_sam.extend("\x00")
    return set(survivors)


def _dict_hits(train, ix):
    """For each document, dictionary factors encountered (even when an S block wins)."""
    hits = {}
    for s in train:
        for (i, ml, dl, kind, x) in _segment(s, ix):
            if dl >= 1:
                f = s[i:i + dl]
                hits[f] = hits.get(f, 0) + 1
    return hits


def _greedy_seed(train, origins=None, progress=True):
    pruned = extract_maximal_candidates(train, origins=origins, progress=progress)
    ix = DictIndex(pruned)
    useful = set()
    for f in _dict_hits(train, ix):
        w = ix.first_match(f)
        if w:
            useful.add(w)
    return pruned & useful


def _cap_seed(S0, train, max_seed_size):
    if max_seed_size is None or len(S0) <= max_seed_size:
        return S0
    ix = DictIndex(S0)
    freq = {w: 0 for w in S0}
    for f, c in _dict_hits(train, ix).items():
        w = ix.first_match(f)
        if w is not None:
            freq[w] += c
    ranked = sorted(S0, key=lambda w: freq[w] * len(w) ** 1.5, reverse=True)
    return set(ranked[:max_seed_size])


# --------------------------------------------------------------------------------------
# API & cross-validation (parallelizable)
# --------------------------------------------------------------------------------------

def _split_data(train_pool, split, fit_size, seed):
    n = len(train_pool)
    idx = list(range(n))
    random.Random(seed + split).shuffle(idx)
    return [train_pool[j] for j in idx[:fit_size]], [train_pool[j] for j in idx[fit_size:]]


_CV = {}      # data shared with workers (fork) to avoid re-serializing the corpus


def _cv_fit_val(split):
    c = _CV
    return _split_data(c["pool"], split, c["fit_size"], c["seed"])


def _cv_prepare(split):
    c = _CV
    fit, _ = _cv_fit_val(split)
    fit_chunks, fit_origins = chunk_corpus(fit, c["max_seq_len"], c["overlap"])
    return _cap_seed(_greedy_seed(fit_chunks, origins=fit_origins, progress=False), fit_chunks, c["max_seed_size"])


def _cv_eval(args):
    split, S0, lam = args
    c = _CV
    fit, val = _cv_fit_val(split)
    fit_chunks, _ = chunk_corpus(fit, c["max_seq_len"], c["overlap"])
    val_chunks, _ = chunk_corpus(val, c["max_seq_len"], c["overlap"])
    S = mdl_prune(S0, fit_chunks, lam, mode=c["prune_mode"], progress=False)
    fset = FastFactorSet(S)
    cost_val = dict_cost_codec(S) + data_cost_codec(val_chunks, fset, sorted(S))
    return cost_val / len(val)


def _make_pool(n_jobs):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        return None
    return ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx)


def select_lambda_cv(train_pool, lambda_grid=None, target_ratio=None, n_splits=8, seed=0,
                     verbose=False, max_seq_len=2000, overlap=100, max_seed_size=None, progress=True,
                     n_jobs=None, prune_mode=-1):
    """n_jobs: number of processes (None = all cores, 1 = sequential). Identical results."""
    if lambda_grid is None:
        lambda_grid = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    target_ratio = target_ratio or 5
    n = len(train_pool)
    fit_size = max(1, round(n / target_ratio))
    assert fit_size < n, "target_ratio too small: no validation file remaining"
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, n_splits * len(lambda_grid)))

    _CV.update(pool=train_pool, fit_size=fit_size, seed=seed, max_seq_len=max_seq_len,
               overlap=overlap, max_seed_size=max_seed_size, prune_mode=prune_mode)
    pool = _make_pool(n_jobs) if n_jobs > 1 else None
    try:
        mapper = pool.map if pool else map
        seeds = list(_progress(mapper(_cv_prepare, range(n_splits)), "Seeds (splits)",
                               total=n_splits, enabled=progress))
        tasks = [(sp, seeds[sp], lam) for sp in range(n_splits) for lam in lambda_grid]
        costs = list(_progress(mapper(_cv_eval, tasks), "Cross-validation",
                               total=len(tasks), enabled=progress))
    finally:
        if pool:
            pool.shutdown()
        _CV.clear()

    results = {}
    for j, lam in enumerate(lambda_grid):
        fold = [costs[sp * len(lambda_grid) + j] for sp in range(n_splits)]    # split order
        results[lam] = sum(fold) / len(fold)
        if verbose:
            print(f"  lambda={lam:<8.2f} mean cost/file (val) = {results[lam]:.2f}")
    best_lambda = min(results, key=results.get)
    return best_lambda, results


_CK = {}      # data shared with workers (fork)


def _ck_eval(k):
    c = _CK
    codec, train = c["codec"], c["train"]
    ctx = codec.build_context(set(c["snaps"][k]), train, train_corpus=train, progress=False)
    hdr = len(ctx.header_bytes)
    if c["header_charge"] == "raw":
        fn = getattr(codec, "get_uncompressed_header_size", None)
        hdr = int(fn(ctx)) if fn is not None else len(zlib.decompress(ctx.header_bytes[4:4 + (int.from_bytes(ctx.header_bytes[:4], "big") & 0x7FFFFFFF)]))
    return hdr + c["lam"] * sum(len(codec.encode_body(x, ctx)) for x in train)


def _refine_search(K, budget, evaluate):
    """Coarse then fine search for the minimum of J(k), k in [0, K-1] (assumed roughly unimodal).
    Coarse grid (~budget/2 points) then, at each turn, evaluate midpoints of the two intervals
    bracketing the best point. For equal J, smaller dictionary (large k) is preferred.
    evaluate(list of indices) -> list of J. Returns {k: J}."""
    vals = {}

    def ev(idxs):
        idxs = [i for i in dict.fromkeys(idxs) if i not in vals]
        if idxs:
            for i, v in zip(idxs, evaluate(idxs)):
                vals[i] = v

    n0 = max(3, min(K, budget // 2))
    ev(sorted({round(i * (K - 1) / (n0 - 1)) for i in range(n0)}))
    while len(vals) < budget:
        ks = sorted(vals)
        best = min(ks, key=lambda k: (vals[k], -k))
        p = ks.index(best)
        lo = ks[p - 1] if p > 0 else best
        hi = ks[p + 1] if p < len(ks) - 1 else best
        cand = [m for m in ((lo + best) // 2, (best + hi) // 2) if m not in vals]
        if not cand:
            break
        ev(cand[:budget - len(vals)])
    return vals


def select_checkpoint_by_codec_cost(snapshots, train, lam, codec, n_checkpoints=25, info=None, n_jobs=None,
                                    search="grid", header_charge="raw"):
    """Chooses, among the dictionaries of the pruning path, the one that minimizes the REAL CODEC COST on the
    training corpus: J(S) = header bytes + lam * sum of body bytes. For ties, keep the smallest dictionary.
    n_jobs processes (None = all cores, 1 = sequential); identical result.

    search="grid"   : ~n_checkpoints regularly spaced checkpoints (v9 behavior).
    search="refine" : coarse then fine search, at most n_checkpoints evaluations (heuristic: assumes
                      J(k) roughly unimodal); fine resolution over the entire path, not just 25 points.
    `snapshots` : any indexable sequence (list or _LazySnapshots)."""
    if search not in ("grid", "refine"):
        raise ValueError(f"unknown search: {search!r} (expected: 'grid' or 'refine')")
    K = len(snapshots)
    if K == 1:
        return snapshots[0]
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    if search == "grid":
        idx = sorted(set(round(i * (K - 1) / (n_checkpoints - 1)) for i in range(n_checkpoints))) if K > n_checkpoints else list(range(K))
        n_jobs = max(1, min(n_jobs, len(idx)))
    else:
        n_jobs = max(1, min(n_jobs, 2 if n_checkpoints >= 6 else 1))   # rounds of 2 evaluations
    if header_charge not in ("raw", "stored"):
        raise ValueError(f"unknown header_charge: {header_charge!r} (expected: 'raw' or 'stored')")
    _CK.update(codec=codec, train=train, snaps=snapshots, lam=lam, header_charge=header_charge)
    pool = _make_pool(n_jobs) if n_jobs > 1 else None
    try:
        mapper = pool.map if pool else map
        if search == "grid":
            vals = dict(zip(idx, mapper(_ck_eval, idx)))
        else:
            vals = _refine_search(K, max(3, n_checkpoints), lambda ii: list(mapper(_ck_eval, ii)))
    finally:
        if pool:
            pool.shutdown()
        _CK.clear()
    best_J, best_k, Js = None, 0, []
    for k in sorted(vals):
        J = vals[k]
        Js.append((k, len(snapshots[k]), J))
        if best_J is None or J <= best_J:
            best_J, best_k = J, k
    if info is not None:
        info.update(checkpoints=Js, chosen_index=best_k, chosen_size=len(snapshots[best_k]), J=best_J,
                    header_charge=header_charge)
    return snapshots[best_k]


def optimize_dictionary_mdl(train, n_total=None, lam=None, max_seq_len=2000, overlap=100, max_seed_size=None,
                            progress=True, select="codec", codec_module="cs_adapt_codec_3_03", n_checkpoints=25,
                            info=None, n_jobs=None, prune_mode=-1, select_search="grid", header_charge="raw"):
    """Greedy seed -> pruning (prune_mode) -> (optional) choice of stopping point by real cost (select_search).
    info (dict) receives t_seed, t_prune, t_select (seconds) and, if select == "codec", checkpoint details."""
    import time
    t0 = time.perf_counter()
    chunks, origins = chunk_corpus(train, max_seq_len, overlap)
    if lam is None:
        assert n_total is not None, "provide n_total or lam"
        lam = n_total / len(train)
    S0 = _cap_seed(_greedy_seed(chunks, origins=origins, progress=progress), chunks, max_seed_size)
    t1 = time.perf_counter()
    if select != "codec":
        S = mdl_prune(S0, chunks, lam, mode=prune_mode, progress=progress)
        if info is not None:
            info.update(t_seed=t1 - t0, t_prune=time.perf_counter() - t1, t_select=0.0)
        return S
    import importlib
    codec = importlib.import_module(codec_module)
    st = {"snapshots": []}
    mdl_prune(S0, chunks, lam, mode=prune_mode, progress=progress, stats=st, n_checkpoints=n_checkpoints)
    t2 = time.perf_counter()
    S = set(select_checkpoint_by_codec_cost(st["snapshots"], chunks, lam, codec, n_checkpoints, info,
                                            n_jobs=n_jobs, search=select_search, header_charge=header_charge))
    if info is not None:
        info.update(t_seed=t1 - t0, t_prune=t2 - t1, t_select=time.perf_counter() - t2)
    return S


def optimize_dictionary_mdl_cv(train_pool, lambda_grid=None, target_ratio=None, n_splits=8, seed=0,
                               verbose=False, max_seq_len=2000, overlap=100, max_seed_size=None, progress=True,
                               n_jobs=None, prune_mode=-1, select_search="grid"):
    best_lambda, cv_results = select_lambda_cv(train_pool, lambda_grid, target_ratio, n_splits, seed, verbose,
                                               max_seq_len, overlap, max_seed_size, progress=progress,
                                               n_jobs=n_jobs, prune_mode=prune_mode)
    S = optimize_dictionary_mdl(train_pool, lam=best_lambda, max_seq_len=max_seq_len, overlap=overlap,
                                max_seed_size=max_seed_size, progress=progress, n_jobs=n_jobs,
                                prune_mode=prune_mode, select_search=select_search)
    return S      # (v7 returned nothing: S was computed then lost)