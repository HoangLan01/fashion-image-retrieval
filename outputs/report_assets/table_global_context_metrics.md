# Global-context intervention (inference-only)

Shuffled cyclically exchanges each block's context vector between samples; uniform replaces learned token weights with equal weights over valid tokens.

| Variant | R@10 | R@50 | ΔR@10 | ΔR@50 | Median rank | Cosine→full | Top-5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full AACL | 16.8302 | 40.7753 | +0.0000 | +0.0000 | 86.0 | 1.0000 | 5.00/5 |
| Shuffled context | 1.2758 | 3.4347 | -15.5545 | -37.3405 | 2080.0 | 0.2223 | 0.09/5 |
| Uniform context | 4.3670 | 13.0520 | -12.4632 | -27.7233 | 951.5 | 0.4819 | 0.35/5 |
