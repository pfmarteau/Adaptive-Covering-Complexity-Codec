
# Adaptive Covering Complexity ($C_S^{\text{adapt}}$)

> **Repository** for reproducing experiments described in the paper *"Adaptive Covering Complexity as a Formal Bridge Between Relative Dictionary Compression and LZ-Style Self-Referencing"*.

## Paper Abstract
Relative dictionary compression and Lempel–Ziv (LZ) parsing exploit two complementary sources of redundancy: a fixed external reference and the growing decoded text. We introduce $C_S^{\text{adapt}}(s)$ to bridge these paradigms, providing both a formal logarithmic law for combinatorial phrase counts and an empirical dual-source codec optimized for short-record collections.


## Overview

Compressing collections of short, highly repetitive records (e.g., microservice JSON logs, genomic reads) suffers from a cold-start penalty under standard Lempel–Ziv methods. This project addresses this challenge through **Adaptive Covering Complexity**, a combinatorial measure where phrases are drawn either from a fixed external dictionary $S$ or from the dynamically decoded target prefix.

### Key Features
* **Dual-Source Architecture:** Combines static global context (minimun description length (MDL)-driven dictionary) and dynamic local context (LZ77-style prefix matches).
* **Bit-Optimal Viterbi Parser:** Replaces simple greedy parsing with a bit-cost-guided dynamic programming parser to minimize final stream size.
* **Linear-Time theoretical scan:** Includes optimal greedy phrase computation in $\mathcal{O}(M+n)$ time via dual suffix automata.
* **Comprehensive Benchmarks:** Scripts and evaluation protocols against general-purpose tools (`zlib`, `zstd`, `Brotli`) and higher-order Markov models.


## Citation
If you use this work in your research, please cite our paper:

```bibtex
@unpublished{marteau:hal-05801771,
  TITLE = {{On the Adaptive Covering Complexity of Strings: Unifying Relative and Self-Referential Compression}},
  AUTHOR = {Marteau, Pierre-Fran{\c c}ois},
  URL = {https://hal.science/hal-05801771},
  NOTE = {working paper or preprint},
  YEAR = {2026},
  MONTH = Oct,
  KEYWORDS = {Adaptive covering complexity Lempel-Ziv parsing relative dictionary compression suffix automata string matching entropy minimum description length (MDL) ; Adaptive covering complexity ; Lempel-Ziv parsing ; relative dictionary compression ; suffix automata ; string matching ; entropy ; minimum description length (MDL)},
  PDF = {https://hal.science/hal-05801771v1/file/adaptive_covering_v15_TIT_consolidated.pdf},
  HAL_ID = {hal-05801771},
  HAL_VERSION = {v1},
}
