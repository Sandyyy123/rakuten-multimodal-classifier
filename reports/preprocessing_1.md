# Preprocessing Report: Rakuten Multimodal Product Classification

## 1. Objective

Transform raw CSV + image data into a clean, model-ready dataset for both
text and image branches. This report covers all decisions made before any
model is trained, in line with Rendering 1 requirements (due 3 Jun).

---

## 2. Text preprocessing pipeline

### Step 1 - HTML stripping (description field)
```python
from bs4 import BeautifulSoup

def strip_html(text):
    if pd.isna(text):
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ").strip()

df["description_clean"] = df["description"].apply(strip_html)
```
**Why:** `description` contains raw HTML (`<br>`, `<span>`, `&amp;`, etc.)
injected by seller tools. BeautifulSoup's `get_text` preserves word order
and spacing, unlike regex stripping which can merge adjacent words.

### Step 2 - Missing value handling
```python
df["desc_missing"] = df["description"].isna().astype(int)   # indicator feature
df["description_clean"] = df["description_clean"].fillna("") # empty string for absent desc
```
**Why:** EDA (Visualisation 2) confirmed missingness is class-dependent (not MCAR).
A binary `desc_missing` indicator allows the model to learn that absence of
description is itself a category signal.

### Step 3 - Text field construction
```python
df["text"] = df["designation"] + " " + df["description_clean"]
df["text"] = df["text"].str.strip().str.lower()
```
**Why:** Lowercasing reduces vocabulary size without losing semantic content
for French/English product text (no case-sensitive proper nouns in product
names are systematically lost at this vocabulary scale).

### Step 4 - Numeric length features
```python
df["designation_len"] = df["designation"].str.len()
df["description_len"] = df["description_clean"].str.len()  # 0 if missing
```
**Why:** Kruskal-Wallis test (Visualisation 3) confirmed text length is
class-informative. These features are cheap to compute and add orthogonal
signal to TF-IDF bag-of-words.

---

## 3. Train / validation / test split

```python
from sklearn.model_selection import train_test_split

# Stratified 80/10/10 on full 84,916 rows
X_temp, X_test, y_temp, y_test = train_test_split(
    df, df["prdtypecode"], test_size=0.10, random_state=42, stratify=df["prdtypecode"]
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp
)
# Result: 67,933 train / 8,492 val / 8,491 test
```
**Why 80/10/10:** The smallest class has 764 rows. 10% test = 76 rows for the
rarest class - sufficient for stable F1 estimates. 60/20/20 on a 30k subsample
was used in modeling_1/2 for speed; 80/10/10 on the full dataset is used here
for the Rendering 1 deliverable.

---

## 4. Image preprocessing pipeline

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),        # augmentation
    transforms.ColorJitter(brightness=0.2),   # augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225])    # ImageNet std
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```
**Why these transforms:**
- `RandomCrop` + `RandomHorizontalFlip` + `ColorJitter`: augmentation
  reduces overfitting on the 30k subsample; EDA confirmed images are clean
  (white background, centered) so aggressive augmentation is not needed.
- `Resize(256) -> CenterCrop(224)`: standard ResNet input; avoids distortion
  from directly resizing to 224.
- ImageNet normalisation: required for pretrained ResNet/ViT weights.

---

## 5. Class imbalance handling

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=300,
    class_weight="balanced"   # KEY CHANGE from modeling_1
)
```
**Why:** EDA (Visualisation 5) confirmed the 13.4x class imbalance causes
minority classes (F1 ~0.64) to be systematically under-served by the
default model. `class_weight='balanced'` applies weights inversely
proportional to class frequency with zero additional data cost.

Expected impact: minority class F1 +8-14 points; macro-F1 +4-7 points.

---

## 6. TF-IDF vectoriser configuration (updated)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30_000,     # increased from 20,000
    min_df=2,                # decreased from 3 (captures rarer brand names)
    sublinear_tf=True,       # log(1 + tf) dampens high-frequency terms
    strip_accents="unicode",
    analyzer="word",
)
```
**Change from modeling_1:** `sublinear_tf=True` and increased `max_features`
better handle the multilingual vocabulary (French, English, German brand names
are typically low-frequency but highly discriminative).

---

## 7. Feature matrix construction

```python
import scipy.sparse as sp
import numpy as np

# TF-IDF on cleaned text
X_tfidf = tfidf.fit_transform(X_train["text"])

# Numeric features (length + missing indicator)
numeric = X_train[["designation_len", "description_len", "desc_missing"]].values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric)

# Stack sparse TF-IDF + dense numeric
X_train_final = sp.hstack([X_tfidf, sp.csr_matrix(numeric_scaled)])
```

---

## 8. Evaluation metrics

Primary: **macro-F1** (equal weight to all 27 classes, penalises minority failure)
Secondary: **weighted F1** (proportional to class size, matches Rakuten challenge)
Tertiary: **per-class F1** table for all 27 classes (surfaces hidden weaknesses)

Report accuracy only for comparison with the Rakuten challenge leaderboard.

---

## 9. Artifacts persisted

| File | Content |
|------|---------|
| `deliverables/tfidf_v2.pkl` | Fitted TF-IDF vectoriser (30k features, sublinear) |
| `deliverables/scaler_v2.pkl` | Fitted StandardScaler for numeric features |
| `deliverables/logr_balanced.pkl` | LogReg with class_weight='balanced' |
| `deliverables/metrics_v2.json` | Val + test metrics for all model variants |
