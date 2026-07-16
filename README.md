# AACL Fashion Image Retrieval

Implementation scaffold for Additive Attention Compositional Learning (AACL) on FashionIQ.

## Vietnamese install guide

For full setup, GPU/CUDA checks, FashionIQ preparation, training commands, and troubleshooting in Vietnamese, see [INSTALL_VI.md](INSTALL_VI.md).

## What is included

- FashionIQ dataset and gallery loaders for `data/fashioniq/{images,captions,image_splits}`.
- Swin image encoder via `timm`, DistilBERT text encoder via HuggingFace Transformers.
- Multi-head additive attention composition module.
- Batch softmax loss plus improved symmetric InfoNCE, label smoothing, dropout.
- GPU-ready training with AMP, gradient clipping, checkpointing, and Recall@K evaluation.
- Dependency-free mock encoders and a synthetic config for CPU smoke tests.

## Expected FashionIQ layout

```text
data/fashioniq/
  images/
  captions/
    cap.dress.train.json
    cap.dress.val.json
  image_splits/
    split.dress.train.json
    split.dress.val.json
```

The loader also accepts a few common filename variants, but the official FashionIQ names above are preferred.

## Commands

Improved config:

```bash
python train.py --config configs/fashioniq.yaml --category dress
python evaluate.py --config configs/fashioniq.yaml --checkpoint outputs/fashioniq_improved/dress/best.pt --category dress
python scripts/retrieve_demo.py --config configs/fashioniq.yaml --checkpoint outputs/fashioniq_improved/dress/best.pt --category dress --query-index 0 --top-k 10 --output outputs/retrieval_demo/dress/query0.jpg
```

Shared L40 config (conservative when roughly 25 GiB is free on the selected GPU):

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/fashioniq_l40_shared.yaml --category dress
```

This run writes to `outputs/fashioniq_improved/l40_shared_seed42/<category>/`. It records
`metrics.csv`, `metrics.jsonl`, the resolved config, run metadata, peak CUDA memory, and atomic
`latest.pt`/`best.pt` checkpoints. The shared config uses `.cache/huggingface/hub` for pretrained Swin and DistilBERT weights, so
the downloaded files are reused without writing to another user's home cache.

Resume an interrupted run with:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/fashioniq_l40_shared.yaml --category dress --resume auto
```

Export aggregate metrics and per-query target ranks for the report:

```bash
python evaluate.py \
  --config configs/fashioniq_l40_shared.yaml \
  --checkpoint outputs/fashioniq_improved/l40_shared_seed42/dress/best.pt \
  --category dress \
  --json-output outputs/fashioniq_improved/l40_shared_seed42/dress/evaluation.json \
  --per-query-output outputs/fashioniq_improved/l40_shared_seed42/dress/per_query.csv
```

Baseline reproduction config:

```bash
python train.py --config configs/fashioniq_baseline.yaml --category dress
```

CPU smoke test without FashionIQ, `timm`, or downloaded HuggingFace weights:

```bash
python scripts/check_setup.py --config configs/synthetic_smoke.yaml
python train.py --config configs/synthetic_smoke.yaml --category dress
python -m unittest discover -s tests
```

Extract post-training additive-attention flow for a fixed counterfactual probe:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/visualize_attention.py \
  --category shirt \
  --probe-id shirt_q01 \
  --device cuda \
  --save-heads
```

This keeps the training API unchanged and writes raw block/head/token weights plus Stage 3,
Stage 4, average-stage, per-head, text-token, and counterfactual visualizations.

Run paired high/low attention occlusion on ten fixed shirt probes:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_occlusion_faithfulness.py \
  --category shirt \
  --num-probes 10 \
  --device cuda
```

Run inference-only global-context interventions on the full shirt validation set:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_context_intervention.py \
  --category shirt \
  --device cuda
```

This encodes the gallery once and compares the unchanged checkpoint with shuffled-context and
uniform-context inference. It writes per-query ranks, paired statistics, a Markdown table, and a
report figure. These interventions test context dependence; they are not substitutes for a
separately retrained architectural ablation.

## Notes

- The default production config assumes training on a CUDA machine.
- The Swin wrapper pools Stage 3 and Stage 4 features to `7x7` each, then concatenates them into 98 image tokens.
- Query images are masked out from the gallery during evaluation when their IDs are present.
