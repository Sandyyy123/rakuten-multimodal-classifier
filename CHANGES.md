# CHANGES - Project #06 Rakuten Multimodal

Reconciliation of manuscript Methods to the executed notebook configuration. The numbers in `deliverables/metrics.json` were produced by the notebook, so the manuscript narrative is brought into alignment with the notebook (the notebook was not modified).

## Methods drift fixes (manuscript)

### Change 1: TF-IDF min document frequency

Section 3.1 (Text branch).

Before: "scikit-learn TfidfVectorizer using ngram_range=(1, 2), max_features=20,000, and min_df=3"

After: "scikit-learn TfidfVectorizer using ngram_range=(1, 2), max_features=20,000, and min_df=2"

Source of truth: `notebooks/build_modeling_nb.py` cell 4 uses `min_df=2`.

### Change 2: Logistic regression solver, max_iter, and class weighting

Section 3.3 (Fusion and head).

Before: "The classifier is a multinomial logistic regression with C=1.0, max_iter=300, and solver lbfgs, trained on the joint matrix with no class weighting at the baseline stage."

After: "The classifier is a multinomial logistic regression with C=1.0, max_iter=1000, solver liblinear, and class_weight='balanced', trained on the joint matrix. The class_weight='balanced' setting reweights training samples by inverse class frequency at fit time and is used because of the 13.4x class imbalance reported in section 2.5; explicit class-balanced loss [31] and focal loss [30] are reserved for a follow-up that targets macro-F1 specifically."

Source of truth: `notebooks/build_modeling_nb.py` cells 6 and 8 use `solver='liblinear'`, `max_iter=1000`, `class_weight='balanced'`.

### Change 3: Image preprocessing pipeline (resize / centre-crop)

Section 3.2 (Image branch).

Before: "decoded with PIL, resized so that the shorter side is 256 pixels, centre-cropped to 224 by 224, converted to a torch tensor, and ImageNet-normalised..."

After: "decoded with PIL, resized directly to 224 by 224 with torchvision transforms.Resize((224, 224)), converted to a torch tensor, and ImageNet-normalised..."

Source of truth: `notebooks/build_modeling_nb.py` uses `transforms.Resize((224, 224))` directly with no shorter-side step and no centre-crop.

### Change 4: Image-feature scaling (L2-norm vs StandardScaler)

Section 3.2 (Image branch).

Before: "Each feature vector is L2-normalised to unit length before fusion."

After: "Each feature vector is rescaled with scikit-learn StandardScaler(with_mean=False) before fusion so that the dense block has unit per-dimension variance while preserving sparsity-compatible zero-centring behaviour."

Section 3.3 (Fusion and head).

Before: "The image branch is L2-normalised before concatenation to prevent the high-dimensional, sparse, longer text vector from dominating the dense image features."

After: "The image branch is rescaled with StandardScaler(with_mean=False) before concatenation to prevent the high-dimensional, sparse, longer text vector from dominating the dense image features while keeping the dense block compatible with a sparse joint matrix."

Source of truth: `notebooks/build_modeling_nb.py` uses `StandardScaler(with_mean=False)` rather than L2 unit-norm.

### Change 5: Evaluation metric mention (top-3 added)

Section 3.4 (Evaluation).

Before: "Top-1 accuracy is reported for completeness."

After: "Top-1 accuracy and top-3 accuracy are reported for completeness."

Rationale: notebook also reports top-3 accuracy; the validator flagged this as a manuscript omission.

## Downstream wording cascades (Abstract, Results, Discussion, Conclusion)

To keep the manuscript internally consistent, every later mention of "L2-normalised" image features, "no class weighting", or the centre-crop pipeline was rewritten to reference StandardScaler-rescaled features and class_weight='balanced'. Specifically:

- Abstract: "L2-normalises the resulting 512-dimensional vector" -> "scales the resulting 512-dimensional vector with a zero-mean-preserving StandardScaler"; "trains a multinomial logistic regression head" -> "trains a class-balanced multinomial logistic regression head".
- Section 3.4: "the only difference ... is the presence or absence of the 512-dimensional L2-normalised image vector" -> "the 512-dimensional StandardScaler-rescaled image vector"; added "class weighting" to the shared-design list.
- Section 4.2: "L2-normalised dense ResNet18 vector" -> "StandardScaler-rescaled dense ResNet18 vector"; added "class weighting" to the shared-design list.
- Section 4.4: two mentions of "L2-normalised generic ImageNet activations" / "dense 512-dimensional L2-normalised vector" rewritten to reference StandardScaler rescaling.
- Section 4.5: "L2-normalised vector concatenation" -> "StandardScaler-rescaled vector concatenation".
- Section 5.2: the L2-normalisation paragraph rewritten to describe per-dimension variance rescaling; "Replacing the L2-normalised concatenation" -> "Replacing the rescaled concatenation".
- Section 6 (Conclusion): "L2-normalised vector concatenation" -> "StandardScaler-rescaled vector concatenation".

## Word count trim

Manuscript word-count target: 4000 to 5000.

Before edits: 5183 words.

After Methods rewrites and downstream cascades: 5209 words (the more explicit Methods phrasings added a few words).

Trimmed redundant prose in Discussion (no Methods detail removed):

- Section 5.1 ("Why text-only TF-IDF reaches 0.75 weighted F1"): three repetitive paragraphs collapsed into two; the brand-token examples and the linear-head argument are preserved, but ceremonial enumeration phrases ("First", "Second", "Third") and one duplicated paragraph were cut.
- Section 5.2 ("Why the multimodal lift is only +0.001"): the first paragraph (two-reasons preamble) was condensed; both reasons retained verbatim in substance.
- Section 5.3 ("What would close the gap"): three multi-sentence paragraphs were tightened; expected-lift numbers, costs, and references all preserved.
- Section 6 (Conclusion): closing "contribution of this paper" sentence trimmed (already stated upstream).

Final word count: 4996 words. Target met (4000 to 5000).

## Verification

- Em-dash scan: 0 hits in `manuscripts/manuscript.md` (live `grep` for U+2014 returned 0).
- Methods now matches `notebooks/build_modeling_nb.py` for the five drifts called out in `validation_report.md` section 6 (min_df, solver, max_iter, class_weight, image preprocessing pipeline, L2 vs StandardScaler).
- Notebook unchanged (per task rule 5).
- Word count read live via `wc -w`: 4996.
