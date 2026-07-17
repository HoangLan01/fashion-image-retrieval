# Occlusion faithfulness

Primary endpoint uses each unoccluded query's top-1 result as the fixed reference.

| Mask ratio | Δsim high | Δsim low | high−low [bootstrap 95% CI] | Δrank high | Δrank low | p (Δsim) |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 0.0015 | -0.0018 | 0.0033 [0.0006; 0.0057] | 0.30 | 0.00 | 0.0391 |
| 20% | 0.0041 | -0.0006 | 0.0047 [-0.0004; 0.0105] | 0.70 | 0.00 | 0.1289 |
| 30% | 0.0053 | -0.0000 | 0.0053 [-0.0025; 0.0134] | 1.00 | 0.20 | 0.2520 |
