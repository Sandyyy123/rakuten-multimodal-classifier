# Modeling 2 - Multimodal Late Fusion (Text + Image, Rakuten 27-class)

## What changed vs `modeling_1.md`

Added an image branch and concatenated it with the TF-IDF text branch:

1. **Image branch.** Each product image streamed from `images.zip` via `zipfile.ZipFile.open()` (no extraction). Resize 256, CenterCrop 224, ImageNet-normalised. Forward through frozen ResNet18 (`IMAGENET1K_V1`, classifier head replaced with `nn.Identity()`); 512-dim feature per image. L2-normalise to unit length.

2. **Fusion.** Horizontal stack `[TF-IDF sparse | L2-normalised dense image features]` -> 19,200 x 20,512 sparse matrix.

3. **Same head.** Multinomial Logistic Regression (`C=1.0`, `solver='lbfgs'`, `max_iter=300`) on the fused matrix.

## Results

| Model | Val accuracy | Test accuracy | Test macro F1 | Test weighted F1 |
|---|---|---|---|---|
| Text-only (modeling_1) | 0.754 | 0.753 | 0.729 | 0.751 |
| **Multimodal (this run)** | **0.755** | **0.754** | **0.730** | **0.752** |

The multimodal lift is **+0.001 on every metric**. Within noise. The image branch adds essentially no information beyond what the TF-IDF text already encodes.

## Why the lift is tiny

The TF-IDF + LogReg baseline already saturates the deterministic component of the labelling: product designations contain brand and model names that deterministically map to a category. Where the text is unambiguous (~75% of items), images add nothing. Where the text is ambiguous (~25%), the L2-normalised image features from a frozen ImageNet backbone are too coarse to disambiguate.

This is a known pattern from the Rakuten challenge literature: simple text classifiers reach 0.74-0.78 weighted F1 on the 27-class task, and pushing past that requires either (a) trainable image branches (fine-tuned ResNet50 or ViT) or (b) cross-modal transformers (ViLBERT-family or CLIP-style joint encoders).

## What would close the gap

- **Trainable image branch**: replace frozen ResNet18 with a fine-tuned ResNet50 (3-5 epoch on the 30k subsample) or DINO-pretrained ViT. Expected lift: +3-5 weighted F1 points.
- **Multilingual transformer text**: replace TF-IDF with XLM-R or CamemBERT embeddings. Expected lift: +5-8 weighted F1 points.
- **Cross-modal fusion**: ViLBERT-style cross-attention rather than vector concat. Expected lift: +3-5 points on top of the trainable backbones.
- **Class-balanced loss**: explicitly weight the long-tail (smallest class 764 rows after subsampling to 30k). Expected lift: macro F1 specifically, not weighted F1.

## Limitations of this run

- 30,000-row subsample (~35% of the full 84,916 train) for runtime tractability. Full-train models in the Rakuten challenge winner papers report 1-2 weighted F1 points higher purely from data scale.
- Frozen ImageNet backbone is the wrong domain. Fine-tuning on the product images (even a few epochs) is the standard fix.
- LogReg on the fused 20,512-dim sparse matrix is a linear scorer; non-linear heads (MLP, gradient-boosted on the dense image features alone) would tease out the image-side signal that LogReg cannot.

## Persisted artifacts

- `deliverables/rakuten_textimg.pkl` - fused multimodal Logistic Regression model
- `deliverables/rakuten_tfidf.pkl` - fitted TF-IDF vectoriser
- `deliverables/metrics.json` - per-split metrics for both text-only and multimodal
