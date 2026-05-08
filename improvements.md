# Improvements - Project #06 Rakuten Multimodal

Role: IMPROVER. Recommendations only. No file modifications.

## Top recommendation

Replace the frozen ImageNet ResNet18 image branch with a CLIP/SigLIP image-text dual encoder used as a feature extractor (no fine-tuning required), and concatenate its image embedding with the existing TF-IDF vector. Rationale: the manuscript itself diagnoses the bottleneck as "frozen ImageNet features are too coarse" because ImageNet-1k category directions do not align with Rakuten's 27 catalogue classes. CLIP-style image features are pretrained on web-scale image-caption pairs that already include catalogue-style product photography and brand imagery, so they encode commerce-relevant directions that ImageNet does not. Drop-in path: `open_clip` ViT-B/32 LAION-2B (or SigLIP base, or multilingual mCLIP) image encoder, 512-d output, L2-normalised, concatenated to the same 20,000-d TF-IDF representation, same logistic regression head. This is a single-day change on CPU and is the cheapest test of whether the gap is fusion-input quality (likely yes) or fusion strategy (likely no). Expected lift: +3 to +6 weighted-F1 points based on Rakuten challenge follow-up papers (Tashu 2022, Bi 2020) and the published CLIP transfer band on retail benchmarks.

## Weaknesses and concrete improvements

### 1. Frozen ImageNet backbone is the wrong domain (HIGH)
Current: ResNet18 IMAGENET1K_V1, classifier head replaced with Identity, no fine-tuning.
Improvement: Either (a) swap to OpenCLIP ViT-B/32 LAION-2B image encoder for catalogue-aligned features (zero training, drop-in), or (b) fine-tune ResNet50 for 3-5 epochs on the 24k stratified train rows with `prdtypecode` as target on a single small GPU (or via xformers on CPU for 10-15 hours). Path (a) first because it is a one-line change in the notebook and tests the hypothesis without GPU access. Use `timm.create_model("vit_base_patch16_clip_224.laion2b", pretrained=True, num_classes=0)` and forward at 224x224.

### 2. Manuscript / code drift on the head and class weighting (HIGH, reproducibility)
Current: manuscript Methods 3.3 states `solver='lbfgs'` with `max_iter=300` and "no class weighting at the baseline stage", but `notebooks/build_modeling_nb.py` cells 6 and 8 use `solver='liblinear'`, `max_iter=1000`, AND `class_weight='balanced'` for both text-only and multimodal runs. This is method drift that a reviewer or grader will catch. Improvement: pick one (the executed code is `liblinear` + balanced weights, which produced the metrics in `metrics.json`) and rewrite the manuscript Methods + modeling_1.md to match the executed notebook exactly. Add an explicit ablation row reporting both `class_weight=None` and `class_weight='balanced'` so the macro-F1 effect of balancing is visible separately from the multimodal effect.

### 3. No multilingual contextual text encoder despite 39 percent non-French content (HIGH)
Current: TF-IDF (1, 2)-grams on lower-cased concatenated designation+description. The manuscript already lists this as the single most likely lift source (+5-8 weighted-F1) but the experiment is not run.
Improvement: Add a third model. Precompute mean-pooled `xlm-roberta-base` or `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` embeddings (768-d or 384-d) on the cleaned text. This is a one-shot CPU pass over 30k rows in 30-60 minutes. Concatenate with the existing TF-IDF (XLM-R alone, then XLM-R + TF-IDF, then XLM-R + TF-IDF + image) and retrain the same LogReg head. Reports four-row ablation: text-TFIDF, text-XLMR, text-TFIDF+image, text-XLMR+image. This converts the paper from "negative result on one fusion" to "complete encoder ablation on a public benchmark", which is much stronger framing.

### 4. No statistical significance test on the +0.001 lift (HIGH)
Current: Manuscript declares +0.001 as "within noise" by inspection, with no test.
Improvement: Run McNemar's paired test on the per-row prediction agreement between text-only and multimodal on the held-out 6,000-row test set. Report the chi-square statistic and p-value. Also bootstrap weighted-F1 over 1,000 resamples of the test set (with replacement, stratified) and report a 95 percent CI for both models and for the delta. This is `scipy.stats.contingency.mcnemar` and a 30-line bootstrap loop. It converts the qualitative claim into a quantitative one and is what a journal reviewer or DataScientest grader will ask for.

### 5. Held-out test set is unused for cross-validation; single split is fragile (MEDIUM)
Current: Single 60/20/20 stratified split with `random_state=42`. Numbers may be brittle to seed.
Improvement: Run stratified 5-fold cross-validation on the 24,000 train+val pool for both models and report mean +/- std weighted-F1 and macro-F1. Keep the 6,000 test as the truly held-out final report, but compute CV on the development pool to characterise seed sensitivity. This costs roughly five extra LogReg fits (a few minutes total) and image features can be reused since they do not depend on the split.

### 6. No long-tail diagnostic; macro-F1 reported only as a single number (MEDIUM)
Current: Manuscript section 4.3 mentions per-class F1 in prose but the manuscript never tables the per-class F1 deltas between text-only and multimodal.
Improvement: Add Table 3 to manuscript: per-class precision, recall, F1, support, and the multimodal-minus-text delta on the held-out test, sorted by support ascending. The interesting finding is the per-class delta on the smallest five classes (60, 2220, 1301, 1940, 1180) where the image branch could in principle help most. Also add a calibration plot (reliability diagram, 10 bins) for both models on the test set since the linear logistic head's probabilities are downstream-relevant for any production scorer.

### 7. Reproducibility: missing `requirements.txt`, missing `src/model_baseline.py` and `src/model_advanced.py`, no `checkpoint.json` (MEDIUM)
Current: `src/` is empty. Only the notebook holds the executable code. The QA rule set expects `src/model_baseline.py` and `src/model_advanced.py` and a `checkpoint.json`.
Improvement: Extract the modeling notebook into two scripts (`src/model_baseline.py` for the text-only run and `src/model_advanced.py` for the multimodal run) so the run is CLI-callable without Jupyter. Pin a `requirements.txt` with the exact versions used (numpy, pandas, scikit-learn, scipy, torch, torchvision, Pillow, nbformat). Write a `checkpoint.json` with the schema the validator expects (project_number, title, methodology, status). This is the single change with the highest validator-PASS impact.

### 8. No image-only baseline; the controlled comparison is incomplete (MEDIUM)
Current: text-only and (text + image), but no image-only. The manuscript's claim "text saturates the deterministic component" is not directly testable without an image-only number.
Improvement: Add a third model: ResNet18 frozen features alone into the same LogReg head (with `class_weight='balanced'` to match). The expected number is roughly 0.40-0.55 weighted-F1 based on related literature. Reporting it lets the manuscript explicitly compare text-alone, image-alone, and both, which is the actual "where does the signal live?" decomposition the paper claims to make.

### 9. Image feature extraction time and lack of caching (LOW)
Current: ResNet18 forward over 30k images is 60-120 minutes single-threaded on CPU and must be re-run if the notebook restarts.
Improvement: Persist the 30,000 by 512 image-feature matrix to `deliverables/resnet18_feats.npz` (or `.pkl`) on first run and reload it on subsequent runs. Cache the file with a `(subsample_seed, backbone_name)` key so a backbone swap (CLIP, ConvNeXt) does not silently load stale features. Adds 8 lines of code, removes the most expensive step on every re-run.

### 10. Presentation HTML weaknesses for a business audience (LOW)
Current: `deliverables/presentation.html` is referenced but not modified to clearly communicate the negative result. (Not opened here, listed because the brief calls for a final report and the presentation is the client-facing artefact.)
Improvement: Lead the deck with a single chart showing four bars on the same y-axis (text-only, multimodal, projected text+CLIP, projected text+CLIP+fine-tuned-vision). Show the +0.001 measured lift next to a +0.05-0.08 projected lift so a non-technical reader sees both that this run was a controlled experiment AND the path to a leaderboard-relevant model. Also add a one-slide "what to read next" with three numbered references for the recommended encoder upgrade.

## Priority summary

- HIGH: 1 (CLIP image branch), 2 (fix code/manuscript drift), 3 (multilingual text encoder), 4 (statistical significance test).
- MEDIUM: 5 (cross-validation), 6 (per-class table + calibration), 7 (src scripts + requirements + checkpoint), 8 (image-only baseline).
- LOW: 9 (cache image features), 10 (presentation framing).

The single highest-leverage change is item 1 (swap ResNet18 frozen features for CLIP-family frozen features) because it directly tests the manuscript's central diagnostic claim ("the inputs are weak, not the fusion") and is achievable in one day on CPU with no GPU dependency. Items 2 and 7 should be bundled with it because they are zero-cost wins for reviewer credibility.
