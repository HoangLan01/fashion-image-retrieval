| Category | Model | Validation | N | R@10 | R@50 | Median target rank | MRR |
|---|---|---|---:|---:|---:|---:|---:|
| shirt | Full train | Full val | 2038 | 16.8302 | 40.7753 | 86.0 | 0.0864 |
| shirt | Lexical holdout | Full val | 2038 | 18.2532 | 39.8430 | 91.0 | 0.0884 |
| shirt | Full train | Lexical val | 167 | 10.1796 | 31.1377 | 144.0 | 0.0583 |
| shirt | Lexical holdout | Lexical val | 167 | 7.1856 | 19.7605 | 186.0 | 0.0384 |
| toptee | Full train | Full val | 1961 | 23.0495 | 50.0765 | 50.0 | 0.1144 |
| toptee | Lexical holdout | Full val | 1961 | 24.4773 | 49.4646 | 53.0 | 0.1201 |
| toptee | Full train | Lexical val | 94 | 13.8298 | 44.6809 | 66.5 | 0.0626 |
| toptee | Lexical holdout | Lexical val | 94 | 12.7660 | 37.2340 | 86.5 | 0.0686 |

| Category | Δ targeted R@10 | 95% CI | Δ targeted R@50 | 95% CI | Rank improved/equal/worsened |
|---|---:|---|---:|---|---:|
| shirt | -2.9940 | [-7.1856, 1.1976] | -11.3772 | [-17.9641, -4.7904] | 61/2/104 |
| toptee | -1.0638 | [-8.5106, 6.3830] | -7.4468 | [-18.0851, 3.1915] | 42/5/47 |

| Category | Surface form | N | Full-train R@10 | Holdout R@10 | ΔR@10 | Full-train R@50 | Holdout R@50 | ΔR@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shirt | `t-shirt` | 54 | 9.2593 | 9.2593 | 0.0000 | 31.4815 | 22.2222 | -9.2593 |
| shirt | `t shirt` | 56 | 16.0714 | 8.9286 | -7.1429 | 35.7143 | 19.6429 | -16.0714 |
| shirt | `tshirt` | 22 | 0.0000 | 0.0000 | 0.0000 | 13.6364 | 0.0000 | -13.6364 |
| shirt | `tee` | 42 | 7.1429 | 4.7619 | -2.3810 | 28.5714 | 23.8095 | -4.7619 |
| toptee | `t-shirt` | 29 | 13.7931 | 10.3448 | -3.4483 | 41.3793 | 41.3793 | 0.0000 |
| toptee | `t shirt` | 10 | 10.0000 | 30.0000 | 20.0000 | 60.0000 | 60.0000 | 0.0000 |
| toptee | `tshirt` | 8 | 25.0000 | 25.0000 | 0.0000 | 50.0000 | 50.0000 | 0.0000 |
| toptee | `tee` | 49 | 12.2449 | 8.1633 | -4.0816 | 40.8163 | 28.5714 | -12.2449 |
