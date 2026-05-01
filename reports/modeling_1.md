# Modeling 1 - Text-only TF-IDF + Logistic Regression (Rakuten 27-class)

## Task

27-class product classification on the Rakuten France challenge dataset. Predict `prdtypecode` from `designation + description` (multilingual, mostly French).

## Sample

Stratified subsample of 30,000 rows from 84,916 training rows. Stratified 60 / 20 / 20 train / val / test split:

- Train: 19,200
- Val: 4,800
- Test: 6,000

Stratification preserves the 27-class distribution including the long tail (smallest class 1180 with proportional 270 train rows).

## Pipeline

1. Concatenate `designation + description`, lower-case. Missing description (35% per EDA) replaced with empty string.
2. TF-IDF vectorise with 1-2 grams, `max_features=20,000`, `min_df=3`. Result: 19,200 x 20,000 sparse matrix.
3. Multinomial Logistic Regression with `solver='lbfgs'`, `C=1.0`, `max_iter=300`.

## Results

| Split | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Validation | 0.754 | 0.725 | 0.750 |
| Test | **0.753** | **0.729** | **0.751** |

Text-only TF-IDF reaches **0.75 weighted F1** on this 30k subsample. Validation and test are tightly aligned (within 0.005 on every metric), suggesting the LogReg is well-regularised at `C=1.0`.

## Why this baseline is strong

Product `designation` is highly discriminative for category: the brand or model name in the designation often deterministically maps to a product category (a "Funko Pop" goes to collectables, "Lego" goes to toys, etc.). Bigrams capture short brand-product collocations. The TF-IDF + LogReg recipe sets a high baseline that purely visual approaches struggle to match on this dataset.

## Configuration details

- Vectoriser: `TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=3)`
- Classifier: `LogisticRegression(C=1.0, max_iter=300, solver='lbfgs')`
- Random seed: `random_state=42` for the split
- Stratified split: `train_test_split(..., stratify=df['target'], random_state=42)` applied twice (test, then val from train)

## What modeling_2 will add

The image branch (frozen ResNet18 features) and late-fusion concatenation. Question: does image add information beyond what the text already encodes? See `modeling_2.md` for the answer.

## Persisted artifacts

- `deliverables/rakuten_tfidf.pkl` - fitted TF-IDF vectoriser
- `deliverables/metrics.json` - per-split metrics for both text-only and multimodal models
