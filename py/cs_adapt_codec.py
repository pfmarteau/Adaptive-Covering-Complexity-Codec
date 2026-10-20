from __future__ import annotations
from dataclasses import dataclass
import math
import zlib

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


BLOCKTYPES = ["L", "D", "S"]
ESCAPE_CHAR = None
MAX_ARITH_VAL = 64
ESCAPE_VAL = MAX_ARITH_VAL + 1


# --- FLUX BINAIRE BUFFERISÉ ET VECTORISÉ ---

class BitStream:
    def __init__(self, buffer: bytearray = None):
        self.buffer = buffer if buffer is not None else bytearray()
        self.bit_buf = 0
        self.bits_count = 0
        self.read_bit_pos = 0

    def write_bit(self, bit: int):
        self.bit_buf = (self.bit_buf << 1) | (bit & 1)
        self.bits_count += 1
        if self.bits_count == 8:
            self.buffer.append(self.bit_buf)
            self.bit_buf = 0
            self.bits_count = 0

    def write_bits(self, val: int, n: int):
        for k in range(n - 1, -1, -1):
            self.write_bit((val >> k) & 1)

    def write_varint(self, val: int):
        while True:
            byte = val & 0x7F
            val >>= 7
            if val:
                self.write_bits(byte | 0x80, 8)
            else:
                self.write_bits(byte, 8)
                break

    def write_elias_gamma(self, x: int):
        assert x >= 1
        b_len = x.bit_length()
        for _ in range(b_len - 1):
            self.write_bit(0)
        self.write_bits(x, b_len)

    def flush(self):
        if self.bits_count > 0:
            self.buffer.append((self.bit_buf << (8 - self.bits_count)) & 0xFF)
            self.bit_buf = 0
            self.bits_count = 0

    def read_bit(self) -> int:
        byte_idx = self.read_bit_pos >> 3
        if byte_idx >= len(self.buffer):
            return 0
        bit_idx = 7 - (self.read_bit_pos & 7)
        self.read_bit_pos += 1
        return (self.buffer[byte_idx] >> bit_idx) & 1

    def read_bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | self.read_bit()
        return v

    def read_varint(self) -> int:
        shift, res = 0, 0
        while True:
            b = self.read_bits(8)
            res |= (b & 0x7F) << shift
            if not (b & 0x80):
                return res
            shift += 7

    def read_elias_gamma(self) -> int:
        zeros = 0
        while self.read_bit() == 0:
            zeros += 1
        if zeros == 0:
            return 1
        return (1 << zeros) | self.read_bits(zeros)


BitReader = BitStream
BitWriter = BitStream


# --- CODEUR / DÉCODEUR ARITHMÉTIQUE ---

class UnifiedArithmeticEncoder:
    def __init__(self, stream: BitStream):
        self.stream = stream
        self.low = 0
        self.high = 0xFFFFFFFF
        self.underflow = 0

    def encode_symbol(self, sym: int, cum_freq: list[int], total_freq: int):
        range_val = self.high - self.low + 1
        self.high = self.low + (range_val * cum_freq[sym + 1]) // total_freq - 1
        self.low = self.low + (range_val * cum_freq[sym]) // total_freq

        while True:
            if self.high < 0x80000000:
                self._emit(0)
            elif self.low >= 0x80000000:
                self._emit(1)
                self.low -= 0x80000000
                self.high -= 0x80000000
            elif self.low >= 0x40000000 and self.high < 0xC0000000:
                self.underflow += 1
                self.low -= 0x40000000
                self.high -= 0x40000000
            else:
                break
            self.low = (self.low << 1) & 0xFFFFFFFF
            self.high = ((self.high << 1) | 1) & 0xFFFFFFFF

    def _emit(self, bit: int):
        self.stream.write_bit(bit)
        while self.underflow > 0:
            self.stream.write_bit(1 - bit)
            self.underflow -= 1

    def finish(self):
        self.underflow += 1
        if self.low < 0x40000000:
            self._emit(0)
        else:
            self._emit(1)


class UnifiedArithmeticDecoder:
    def __init__(self, stream: BitStream):
        self.stream = stream
        self.low = 0
        self.high = 0xFFFFFFFF
        self.value = stream.read_bits(32)

    def decode_symbol(self, cum_freq: list[int], total_freq: int) -> int:
        range_val = self.high - self.low + 1
        target = ((self.value - self.low + 1) * total_freq - 1) // range_val

        sym = 0
        low_i, high_i = 0, len(cum_freq) - 2
        while low_i <= high_i:
            mid = (low_i + high_i) // 2
            if cum_freq[mid + 1] <= target:
                low_i = mid + 1
            elif cum_freq[mid] > target:
                high_i = mid - 1
            else:
                sym = mid
                break

        self.high = self.low + (range_val * cum_freq[sym + 1]) // total_freq - 1
        self.low = self.low + (range_val * cum_freq[sym]) // total_freq

        while True:
            if self.high < 0x80000000:
                pass
            elif self.low >= 0x80000000:
                self.value -= 0x80000000
                self.low -= 0x80000000
                self.high -= 0x80000000
            elif self.low >= 0x40000000 and self.high < 0xC0000000:
                self.value -= 0x40000000
                self.low -= 0x40000000
                self.high -= 0x40000000
            else:
                break
            self.low = (self.low << 1) & 0xFFFFFFFF
            self.high = ((self.high << 1) | 1) & 0xFFFFFFFF
            self.value = ((self.value << 1) | self.stream.read_bit()) & 0xFFFFFFFF

        return sym


# --- AUTOMATE ET INDEXATION LZ77 ---

class SuffixAutomaton:
    def __init__(self):
        self.trans, self.link, self.length, self.last = [{}], [-1], [0], 0

    def extend(self, c):
        cur = len(self.trans)
        self.trans.append({})
        self.length.append(self.length[self.last] + 1)
        self.link.append(-1)
        p = self.last

        while p != -1 and c not in self.trans[p]:
            self.trans[p][c] = cur
            p = self.link[p]
        if p == -1:
            self.link[cur] = 0
        else:
            q = self.trans[p][c]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = len(self.trans)
                self.trans.append(dict(self.trans[q]))
                self.length.append(self.length[p] + 1)
                self.link.append(self.link[q])
                while p != -1 and self.trans[p].get(c) == q:
                    self.trans[p][c] = clone
                    p = self.link[p]
                self.link[q] = clone
                self.link[cur] = clone
        self.last = cur

    def all_matches(self, s: str, start: int) -> list[int]:
        matches = []
        state, length, n = 0, 0, len(s)
        i = start
        while i < n and length < 256:
            nxt = self.trans[state].get(s[i])
            if nxt is None:
                break
            state = nxt
            length += 1
            i += 1
            matches.append(length)
        return matches


def build_dict_automaton(S_list: list[str]) -> SuffixAutomaton:
    sam, SEP = SuffixAutomaton(), object()
    for w in S_list:
        for c in w:
            sam.extend(c)
        sam.extend(SEP)
    return sam


def _build_dict_position_index(S_list: list[str], max_match_len: int = 256,
                               min_match_len: int = 1) -> dict:
    pos_index = {}
    for wi, word in enumerate(S_list):
        n = len(word)
        for start in range(n):
            stop = min(n, start + max_match_len)
            for end in range(start + min_match_len, stop + 1):
                key = word[start:end]
                if key not in pos_index:
                    pos_index[key] = (wi, start)
    return pos_index


class LZ77Index:
    """Index de hachage glissant pour remplacer la recherche s.rfind."""
    def __init__(self, min_len: int = 3):
        self.min_len = min_len
        self.pos_map = {}

    def insert(self, s: str, pos: int):
        if pos + self.min_len <= len(s):
            key = s[pos:pos + self.min_len]
            if key not in self.pos_map:
                self.pos_map[key] = []
            self.pos_map[key].append(pos)

    def find_matches(self, s: str, pos: int, max_antecedents: int = 16) -> list[tuple[int, int]]:
        if pos + self.min_len > len(s):
            return []
        key = s[pos:pos + self.min_len]
        indices = self.pos_map.get(key)
        if not indices:
            return []

        import bisect
        upto = bisect.bisect_left(indices, pos)
        if upto == 0:
            return []

        candidates = []
        max_possible = min(len(s) - pos, 256)

        for prev_pos in reversed(indices[max(0, upto - max_antecedents):upto]):
            length = 0
            while length < max_possible and s[prev_pos + length] == s[pos + length]:
                length += 1
            dist = pos - prev_pos
            if length >= self.min_len:
                candidates.append((dist, length))

        if not candidates:
            return []

        candidates.sort(key=lambda x: (-x[1], x[0]))
        pareto = []
        best_dist = float("inf")
        for dist, length in candidates:
            if dist < best_dist:
                pareto.append((dist, length))
                best_dist = dist

        return pareto


class FreqModel:
    def __init__(self, symbols: list, counts: dict = None, laplace: int = 1):
        self.symbols = list(symbols)
        self.sym_to_idx = {s: i for i, s in enumerate(self.symbols)}
        freqs = [laplace] * len(self.symbols)
        if counts:
            for s, c in counts.items():
                if s in self.sym_to_idx:
                    freqs[self.sym_to_idx[s]] += c

        self.cum_freq = [0] * (len(self.symbols) + 1)
        c = 0
        for i, f in enumerate(freqs):
            self.cum_freq[i] = c
            c += f
        self.cum_freq[-1] = c
        self.total_freq = c

        self.log_probs = []
        for i in range(len(self.symbols)):
            f = self.cum_freq[i + 1] - self.cum_freq[i]
            self.log_probs.append(math.log2(self.total_freq / f) if f > 0 else 30.0)

        esc_idx = self.sym_to_idx.get(ESCAPE_VAL)
        if esc_idx is not None:
            f_esc = self.cum_freq[esc_idx + 1] - self.cum_freq[esc_idx]
            self.escape_cost = math.log2(self.total_freq / f_esc) if f_esc > 0 else 30.0
        else:
            self.escape_cost = 30.0

    def get_index(self, sym) -> int:
        return self.sym_to_idx[sym]

    def get_symbol(self, idx: int):
        return self.symbols[idx]


def _compact_symbols(counts: dict, minimum, include_escape: bool = True) -> list:
    observed = {int(k) for k, v in counts.items() if v > 0 and int(k) <= MAX_ARITH_VAL}
    symbols = sorted(observed | set(minimum if isinstance(minimum, (list, tuple, set)) else []))
    if isinstance(minimum, int):
        symbols = sorted(set(symbols) | {minimum})
    if include_escape:
        symbols.append(ESCAPE_VAL)
    return sorted(set(symbols))


@dataclass
class Context:
    S_list: list
    dict_sam: SuffixAutomaton | None
    pos_index: dict
    alphabet: list
    alphabet_set: set
    blocktype_models: dict[str, FreqModel]
    literal_model: FreqModel
    wordidx_model: FreqModel | None
    length_D_model: FreqModel
    length_S_model: FreqModel
    dist_model: FreqModel
    offset_model: FreqModel | None
    header_bytes: bytes
    bm_costs: dict[str, dict[str, float]] = None


def _val_cost(model, val, is_offset=False):
    if model is None:
        return math.log2(max(2, val + 1))
    if val <= MAX_ARITH_VAL:
        idx = model.sym_to_idx.get(val)
        if idx is not None:
            return model.log_probs[idx]
    payload = val + 1 if is_offset else val
    b_len = payload.bit_length() if payload > 0 else 1
    return model.escape_cost + (2 * b_len - 1)


# --- PARSING PAR PROGRAMMATION DYNAMIQUE OPTIMISÉ ---

def _optimal_parse(
    s: str,
    S_list: list,
    dict_sam: SuffixAutomaton | None,
    pos_index: dict,
    cost_ctx: Context | None = None,
) -> list:
    n = len(s)
    if n == 0:
        return []
    states = BLOCKTYPES
    inf = float("inf")

    dp = {st: [inf] * (n + 1) for st in states}
    back = {st: [None] * (n + 1) for st in states}
    dp_start = [inf] * (n + 1)
    dp_start[0] = 0.0

    n_s = len(S_list)
    mtf_state_start = list(range(n_s))
    mtf_state = {st: [None] * (n + 1) for st in states}

    if cost_ctx is not None:
        bm_costs = cost_ctx.bm_costs
        lm = cost_ctx.literal_model
        alpha_set = cost_ctx.alphabet_set
        ld_m = cost_ctx.length_D_model
        ls_m = cost_ctx.length_S_model
        dist_m = cost_ctx.dist_model
        off_m = cost_ctx.offset_model
        w_m = cost_ctx.wordidx_model
    else:
        bm_costs = lm = alpha_set = ld_m = ls_m = dist_m = off_m = w_m = None

    lz_index = LZ77Index(min_len=3)
    all_prevs = ["START"] + states

    for i in range(n):
        min_prev = dp_start[i]
        for st in states:
            if dp[st][i] < min_prev:
                min_prev = dp[st][i]

        if min_prev == inf:
            lz_index.insert(s, i)
            continue

        char_i = s[i]

        # 1. Mode LITTÉRAL (L)
        if lm and alpha_set and char_i in alpha_set:
            idx = lm.sym_to_idx.get(char_i, -1)
            lit_c = lm.log_probs[idx] if idx >= 0 else 12.0
        elif lm:
            esc_idx = lm.sym_to_idx.get(ESCAPE_CHAR, -1)
            esc_c = lm.log_probs[esc_idx] if esc_idx >= 0 else 12.0
            char_code = ord(char_i)
            varint_bits = (char_code.bit_length() // 7 + 1) * 8
            lit_c = esc_c + varint_bits
        else:
            lit_c = 8.0

        for prev in all_prevs:
            cur = dp_start[i] if prev == "START" else dp[prev][i]
            if cur == inf:
                continue

            cur_mtf = mtf_state_start if prev == "START" else mtf_state[prev][i]
            if cur_mtf is None:
                cur_mtf = mtf_state_start

            bt_c = bm_costs[prev]["L"] if bm_costs else 2.5
            c = cur + bt_c + lit_c
            if c < dp["L"][i + 1]:
                dp["L"][i + 1] = c
                back["L"][i + 1] = (prev, ("L", char_i), i)
                mtf_state["L"][i + 1] = cur_mtf

        # 2. Mode DICTIONNAIRE (D)
        if dict_sam is not None:
            matches = dict_sam.all_matches(s, i)
            if matches:
                max_m = matches[-1]
                for m_len in range(3, max_m + 1):
                    factor = s[i:i + m_len]
                    item = pos_index.get(factor)
                    if item is None:
                        continue
                    wi, off = item

                    j = i + m_len
                    if j <= n:
                        for prev in all_prevs:
                            cur = dp_start[i] if prev == "START" else dp[prev][i]
                            if cur == inf:
                                continue

                            cur_mtf = mtf_state_start if prev == "START" else mtf_state[prev][i]
                            if cur_mtf is None:
                                cur_mtf = mtf_state_start

                            rank = cur_mtf.index(wi) if wi in cur_mtf else wi
                            w_cost = _val_cost(w_m, rank) if w_m else math.log2(max(2, n_s))
                            d_block_cost = _val_cost(ld_m, m_len) + w_cost + _val_cost(off_m, off, is_offset=True)

                            bt_c = bm_costs[prev]["D"] if bm_costs else 2.5
                            c = cur + bt_c + d_block_cost
                            if c < dp["D"][j]:
                                dp["D"][j] = c
                                back["D"][j] = (prev, ("D", wi, off, m_len), i)
                                new_mtf = list(cur_mtf)
                                if wi in new_mtf:
                                    new_mtf.insert(0, new_mtf.pop(new_mtf.index(wi)))
                                mtf_state["D"][j] = new_mtf

        # 3. Mode AUTO-RÉFÉRENCE (S)
        lz_matches = lz_index.find_matches(s, i)
        for dist, slen in lz_matches:
            for cand_len in range(3, slen + 1):
                s_block_cost = _val_cost(ls_m, cand_len) + _val_cost(dist_m, dist)
                j = i + cand_len
                if j <= n:
                    for prev in all_prevs:
                        cur = dp_start[i] if prev == "START" else dp[prev][i]
                        if cur == inf:
                            continue

                        cur_mtf = mtf_state_start if prev == "START" else mtf_state[prev][i]
                        if cur_mtf is None:
                            cur_mtf = mtf_state_start

                        bt_c = bm_costs[prev]["S"] if bm_costs else 2.5
                        c = cur + bt_c + s_block_cost
                        if c < dp["S"][j]:
                            dp["S"][j] = c
                            back["S"][j] = (prev, ("S", dist, cand_len), i)
                            mtf_state["S"][j] = cur_mtf

        lz_index.insert(s, i)

    best_st = min(states, key=lambda st: dp[st][n])
    if dp[best_st][n] == inf:
        raise ValueError("Parsing impossible")

    out = []
    pos = n
    si = best_st
    while pos > 0:
        prev_si, block, prev_pos = back[si][pos]
        out.append(block)
        pos = prev_pos
        si = prev_si
    out.reverse()
    return out


def _add_block_counts(counts, blocks, S_list):
    bt_counts, lit_counts, wi_counts, len_D_counts, len_S_counts, dist_counts, off_counts = counts
    prev_bt = "START"
    mtf = list(range(len(S_list)))
    for block in blocks:
        bt = block[0]
        bt_counts[prev_bt][bt] = bt_counts[prev_bt].get(bt, 0) + 1
        prev_bt = bt
        if bt == "L":
            ch = block[1]
            lit_counts[ch] = lit_counts.get(ch, 0) + 1
        elif bt == "D":
            wi, off, length = block[1], block[2], block[3]
            idx = mtf.index(wi)
            wi_counts[idx] = wi_counts.get(idx, 0) + 1
            mtf.insert(0, mtf.pop(idx))
            off_s = off if off <= MAX_ARITH_VAL else ESCAPE_VAL
            off_counts[off_s] = off_counts.get(off_s, 0) + 1
            l_s = length if length <= MAX_ARITH_VAL else ESCAPE_VAL
            len_D_counts[l_s] = len_D_counts.get(l_s, 0) + 1
        else:
            dist, length = block[1], block[2]
            l_s = length if length <= MAX_ARITH_VAL else ESCAPE_VAL
            d_s = dist if dist <= MAX_ARITH_VAL else ESCAPE_VAL
            len_S_counts[l_s] = len_S_counts.get(l_s, 0) + 1
            dist_counts[d_s] = dist_counts.get(d_s, 0) + 1


def _make_context_from_counts(S_list, dict_sam, pos_index, alphabet, counts):
    bt_counts, lit_counts, wi_counts, len_D_counts, len_S_counts, dist_counts, off_counts = counts

    len_D_syms = _compact_symbols(len_D_counts, [1])
    len_S_syms = _compact_symbols(len_S_counts, [1])
    dist_syms = _compact_symbols(dist_counts, [1])
    off_syms = _compact_symbols(off_counts, [0]) if S_list else None
    wi_syms = list(range(len(S_list))) if S_list else None

    bm_models = {prev: FreqModel(BLOCKTYPES, bt_counts[prev]) for prev in BLOCKTYPES + ["START"]}
    
    bm_costs = {}
    for prev, m in bm_models.items():
        bm_costs[prev] = {
            bt: m.log_probs[m.sym_to_idx[bt]] if bt in m.sym_to_idx else 30.0
            for bt in BLOCKTYPES
        }

    return Context(
        S_list, dict_sam, pos_index, alphabet, set(alphabet),
        bm_models,
        FreqModel(alphabet + [ESCAPE_CHAR], lit_counts),
        FreqModel(wi_syms, wi_counts) if S_list else None,
        FreqModel(len_D_syms, len_D_counts),
        FreqModel(len_S_syms, len_S_counts),
        FreqModel(dist_syms, dist_counts),
        FreqModel(off_syms, off_counts) if S_list else None,
        b"",
        bm_costs
    )


def build_context(S: set, alphabet_source, train_corpus=None, progress=True,
                  refine_passes: int = 3) -> Context:
    S_list = sorted(S)
    dict_sam = build_dict_automaton(S_list) if S_list else None
    pos_index = _build_dict_position_index(S_list)

    if isinstance(alphabet_source, str):
        alphabet_source = [alphabet_source]
    alphabet = sorted(set(c for s in alphabet_source for c in s))

    train_corpus = list(train_corpus or [])
    source = alphabet_source[0]
    corpus = train_corpus + ([source] if source not in train_corpus else [])

    def fresh_counts():
        return ({prev: {} for prev in BLOCKTYPES + ["START"]}, {}, {}, {}, {}, {}, {})

    counts = fresh_counts()
    for s in tqdm(corpus, desc="Analyse", disable=not progress):
        _add_block_counts(counts, _optimal_parse(s, S_list, dict_sam, pos_index), S_list)
    ctx = _make_context_from_counts(S_list, dict_sam, pos_index, alphabet, counts)

    for it in range(refine_passes):
        counts = fresh_counts()
        for s in tqdm(corpus, desc=f"Raffinement {it + 1}/{refine_passes}", disable=not progress):
            _add_block_counts(counts, _optimal_parse(s, S_list, dict_sam, pos_index, ctx), S_list)
        ctx = _make_context_from_counts(S_list, dict_sam, pos_index, alphabet, counts)

    ctx.header_bytes = write_context_header(ctx)
    return ctx


# --- SÉRIALISATION / DÉSÉRIALISATION DE L'EN-TÊTE ---

def write_context_header(ctx: Context) -> bytes:
    stream = BitStream()

    stream.write_varint(len(ctx.alphabet))
    for ch in ctx.alphabet:
        stream.write_varint(ord(ch))

    stream.write_varint(len(ctx.S_list))
    for w in ctx.S_list:
        stream.write_varint(len(w))
        for ch in w:
            stream.write_varint(ord(ch))

    def write_model(model: FreqModel):
        stream.write_varint(len(model.symbols))
        for sym in model.symbols:
            stream.write_varint(int(sym) if sym is not None else 0xFFFFFFFF)
        for i in range(len(model.symbols)):
            freq = model.cum_freq[i + 1] - model.cum_freq[i]
            stream.write_varint(freq)

    for prev in BLOCKTYPES + ["START"]:
        for i in range(3):
            stream.write_varint(ctx.blocktype_models[prev].cum_freq[i + 1] -
                                ctx.blocktype_models[prev].cum_freq[i])

    stream.write_varint(len(ctx.literal_model.symbols))
    for sym in ctx.literal_model.symbols:
        stream.write_varint(0xFFFFFFFF if sym is None else ord(sym))
    for i in range(len(ctx.literal_model.symbols)):
        stream.write_varint(ctx.literal_model.cum_freq[i + 1] - ctx.literal_model.cum_freq[i])

    if ctx.S_list:
        write_model(ctx.wordidx_model)

    write_model(ctx.length_D_model)
    write_model(ctx.length_S_model)
    write_model(ctx.dist_model)

    if ctx.S_list:
        write_model(ctx.offset_model)

    stream.flush()
    raw = bytes(stream.buffer)
    hdr_data = zlib.compress(raw, level=9)
    return (0x80000000 | len(hdr_data)).to_bytes(4, byteorder="big") + hdr_data


def read_context_header(file_bytes: bytes) -> tuple[Context, bytes]:
    header_tag = int.from_bytes(file_bytes[:4], byteorder="big")
    compressed = bool(header_tag & 0x80000000)
    header_len = header_tag & 0x7FFFFFFF
    header_payload = file_bytes[4:4 + header_len]
    if compressed:
        header_payload = zlib.decompress(header_payload)
    body_bytes = file_bytes[4 + header_len:]

    stream = BitStream(bytearray(header_payload))

    n_alpha = stream.read_varint()
    alphabet = [chr(stream.read_varint()) for _ in range(n_alpha)]

    n_S = stream.read_varint()
    S_list = []
    for _ in range(n_S):
        w_len = stream.read_varint()
        S_list.append("".join(chr(stream.read_varint()) for _ in range(w_len)))

    dict_sam = build_dict_automaton(S_list) if S_list else None
    pos_index = _build_dict_position_index(S_list)

    def read_fixed_model(symbols):
        m = FreqModel(symbols)
        m.cum_freq = [0] * (len(symbols) + 1)
        c = 0
        for i in range(len(symbols)):
            f = stream.read_varint()
            m.cum_freq[i] = c
            c += f
        m.cum_freq[-1] = c
        m.total_freq = c
        m.log_probs = []
        for i in range(len(symbols)):
            freq = m.cum_freq[i + 1] - m.cum_freq[i]
            m.log_probs.append(math.log2(m.total_freq / freq) if freq > 0 else 30.0)
        return m

    def read_model():
        ns = stream.read_varint()
        symbols = []
        for _ in range(ns):
            val = stream.read_varint()
            symbols.append(None if val == 0xFFFFFFFF else val)
        m = FreqModel(symbols)
        m.cum_freq = [0] * (ns + 1)
        c = 0
        for i in range(ns):
            f = stream.read_varint()
            m.cum_freq[i] = c
            c += f
        m.cum_freq[-1] = c
        m.total_freq = c
        m.log_probs = []
        for i in range(ns):
            freq = m.cum_freq[i + 1] - m.cum_freq[i]
            m.log_probs.append(math.log2(m.total_freq / freq) if freq > 0 else 30.0)
        return m

    blocktype_models = {prev: read_fixed_model(BLOCKTYPES) for prev in BLOCKTYPES + ["START"]}
    bm_costs = {}
    for prev, m in blocktype_models.items():
        bm_costs[prev] = {
            bt: m.log_probs[m.sym_to_idx[bt]] if bt in m.sym_to_idx else 30.0
            for bt in BLOCKTYPES
        }

    n_lit_syms = stream.read_varint()
    lit_symbols = []
    for _ in range(n_lit_syms):
        val = stream.read_varint()
        lit_symbols.append(None if val == 0xFFFFFFFF else chr(val))
    literal_model = read_fixed_model(lit_symbols)

    wordidx_model = read_model() if S_list else None
    length_D_model = read_model()
    length_S_model = read_model()
    dist_model = read_model()
    offset_model = read_model() if S_list else None

    ctx = Context(
        S_list, dict_sam, pos_index, alphabet, set(alphabet),
        blocktype_models, literal_model, wordidx_model,
        length_D_model, length_S_model, dist_model, offset_model,
        file_bytes[:4 + header_len],
        bm_costs
    )
    return ctx, body_bytes


# --- ENCODAGE ET DÉCODAGE DU CORPS ---

def _encode_val(val: int, model: FreqModel, enc: UnifiedArithmeticEncoder, stream: BitStream, is_offset=False):
    if val <= MAX_ARITH_VAL and val in model.sym_to_idx:
        enc.encode_symbol(model.get_index(val), model.cum_freq, model.total_freq)
    else:
        enc.encode_symbol(model.get_index(ESCAPE_VAL), model.cum_freq, model.total_freq)
        stream.write_elias_gamma(val + 1 if is_offset else val)


def _decode_val(model: FreqModel, dec: UnifiedArithmeticDecoder, stream: BitStream, is_offset=False) -> int:
    sym = model.get_symbol(dec.decode_symbol(model.cum_freq, model.total_freq))
    if sym == ESCAPE_VAL:
        v = stream.read_elias_gamma()
        return v - 1 if is_offset else v
    return sym


def encode_body(s: str, ctx: Context) -> bytes:
    arithmetic = BitStream()
    side = BitStream()

    enc = UnifiedArithmeticEncoder(arithmetic)
    mtf = list(range(len(ctx.S_list)))
    prev_bt = "START"

    plan = _optimal_parse(s, ctx.S_list, ctx.dict_sam, ctx.pos_index, ctx)
    for block in plan:
        bt = block[0]
        bt_m = ctx.blocktype_models[prev_bt]
        enc.encode_symbol(bt_m.get_index(bt), bt_m.cum_freq, bt_m.total_freq)
        prev_bt = bt

        if bt == "L":
            ch = block[1]
            if ch in ctx.alphabet_set:
                enc.encode_symbol(ctx.literal_model.get_index(ch), ctx.literal_model.cum_freq, ctx.literal_model.total_freq)
            else:
                enc.encode_symbol(ctx.literal_model.get_index(ESCAPE_CHAR), ctx.literal_model.cum_freq, ctx.literal_model.total_freq)
                side.write_varint(ord(ch))
        elif bt == "D":
            _, wi, off, length = block
            _encode_val(length, ctx.length_D_model, enc, side)

            idx = mtf.index(wi)
            enc.encode_symbol(ctx.wordidx_model.get_index(idx), ctx.wordidx_model.cum_freq, ctx.wordidx_model.total_freq)
            mtf.insert(0, mtf.pop(idx))

            _encode_val(off, ctx.offset_model, enc, side, is_offset=True)
        else:
            _, dist, length = block
            _encode_val(length, ctx.length_S_model, enc, side)
            _encode_val(dist, ctx.dist_model, enc, side)

    enc.finish()
    arithmetic.flush()
    side.flush()

    a = bytes(arithmetic.buffer)
    b = bytes(side.buffer)

    meta = BitStream()
    meta.write_varint(len(s))
    meta.write_varint(len(a))
    meta.write_varint(len(b))
    meta.flush()
    return bytes(meta.buffer) + a + b


def decode_body(body_bytes: bytes, ctx: Context) -> str:
    meta = BitStream(bytearray(body_bytes))
    n = meta.read_varint()
    arithmetic_len = meta.read_varint()
    side_len = meta.read_varint()

    meta_pos = (meta.read_bit_pos + 7) // 8
    arithmetic_bytes = body_bytes[meta_pos:meta_pos + arithmetic_len]
    side_bytes = body_bytes[meta_pos + arithmetic_len:meta_pos + arithmetic_len + side_len]

    arithmetic = BitStream(bytearray(arithmetic_bytes))
    side = BitStream(bytearray(side_bytes))
    dec = UnifiedArithmeticDecoder(arithmetic)

    out = []
    mtf = list(range(len(ctx.S_list)))
    prev_bt = "START"

    while len(out) < n:
        bt_m = ctx.blocktype_models[prev_bt]
        bt = bt_m.get_symbol(dec.decode_symbol(bt_m.cum_freq, bt_m.total_freq))
        prev_bt = bt

        if bt == "L":
            sym = ctx.literal_model.get_symbol(dec.decode_symbol(ctx.literal_model.cum_freq, ctx.literal_model.total_freq))
            out.append(chr(side.read_varint()) if sym == ESCAPE_CHAR else sym)
        elif bt == "D":
            length = _decode_val(ctx.length_D_model, dec, side)
            idx_symbol = dec.decode_symbol(ctx.wordidx_model.cum_freq, ctx.wordidx_model.total_freq)
            mtf_pos = ctx.wordidx_model.get_symbol(idx_symbol)
            if mtf_pos < 0 or mtf_pos >= len(mtf):
                raise ValueError("Position MTF invalide")
            wi = mtf.pop(mtf_pos)
            mtf.insert(0, wi)

            off = _decode_val(ctx.offset_model, dec, side, is_offset=True)
            if off < 0 or off + length > len(ctx.S_list[wi]):
                raise ValueError("Référence dictionnaire invalide")
            out.extend(ctx.S_list[wi][off:off + length])
        else:
            length = _decode_val(ctx.length_S_model, dec, side)
            dist = _decode_val(ctx.dist_model, dec, side)
            if dist <= 0 or dist > len(out):
                raise ValueError("Distance d'auto-référence invalide")
            start = len(out) - dist
            for i in range(length):
                out.append(out[start + i])

        if len(out) > n:
            raise ValueError("Longueur décodée supérieure à la longueur annoncée")

    return "".join(out)


def corpus_compressed_size(corpus: list[str], S: set, alphabet_source=None, train_corpus=None, progress=True) -> int:
    ctx = build_context(
        S, alphabet_source if alphabet_source is not None else corpus,
        train_corpus=train_corpus, progress=progress
    )

    total_size = len(ctx.header_bytes)
    iterator = tqdm(corpus, desc="Encoding", disable=not progress)
    for s in iterator:
        total_size += len(encode_body(s, ctx))
    return total_size


# --- FONCTIONS D'ANALYSE ET DE VENTILATION DES COÛTS ---

def get_dictionary_raw_size(ctx: Context) -> int:
    """Retourne la taille brute cumulée (en octets) de tous les mots du dictionnaire S."""
    return sum(len(w.encode("latin-1")) for w in ctx.S_list)


def get_uncompressed_header_size(ctx: Context) -> int:
    """Retourne la taille (en octets) de l'en-tête (alphabet, dictionnaire, modèles) avant compression zlib."""
    header_tag = int.from_bytes(ctx.header_bytes[:4], byteorder="big")
    header_len = header_tag & 0x7FFFFFFF
    compressed_payload = ctx.header_bytes[4:4 + header_len]
    return len(zlib.decompress(compressed_payload))


def get_codec_cost_breakdown(corpus: list[str], S: set, alphabet_source=None, train_corpus=None, progress=True) -> dict:
    """Fournit une ventilation complète des coûts d'encodage (raw, en-tête décompressé/compressé, payload)."""
    ctx = build_context(
        S, alphabet_source if alphabet_source is not None else corpus,
        train_corpus=train_corpus, progress=progress
    )

    payload_size = 0
    iterator = tqdm(corpus, desc="Calcul du payload", disable=not progress)
    for s in iterator:
        payload_size += len(encode_body(s, ctx))

    raw_dict_bytes = get_dictionary_raw_size(ctx)
    uncomp_hdr_bytes = get_uncompressed_header_size(ctx)
    comp_hdr_bytes = len(ctx.header_bytes)

    return {
        "raw_dictionary_bytes": raw_dict_bytes,
        "uncompressed_header_bytes": uncomp_hdr_bytes,
        "compressed_header_bytes": comp_hdr_bytes,
        "payload_bytes": payload_size,
        "total_compressed_bytes": comp_hdr_bytes + payload_size,
        "number_of_words": len(ctx.S_list)
    }