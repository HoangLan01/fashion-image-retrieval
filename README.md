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

## Notes

- The default production config assumes training on a CUDA machine.
- The Swin wrapper pools Stage 3 and Stage 4 features to `7x7` each, then concatenates them into 98 image tokens.
- Query images are masked out from the gallery during evaluation when their IDs are present.
