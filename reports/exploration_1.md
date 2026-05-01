# Exploration Report 1: Rakuten Multimodal Product Classification

## 1. Project context

This is the Rakuten France Multimodal Product Data Classification challenge
hosted on `challengedata.ens.fr/challenges/35`. The task: predict
`prdtypecode` (product type code in Rakuten's internal taxonomy) given a
product's text fields (designation, description) and a single product image.
The full Rakuten catalog has thousands of leaf categories; the challenge
exposes a curated subset with 27 type codes covering the main verticals.

## 2. Files and schema

| File | Rows | Columns |
|------|------|---------|
| `X_train_update.csv` | 84,916 | `designation`, `description`, `productid`, `imageid` |
| `X_test_update.csv`  | 13,812 | same as train (no label) |
| `Y_train_CVw08PX.csv`| 84,916 | `prdtypecode` |
| `images.zip`         | ~99k JPGs (2.5 GB) | `images/image_train/...`, `images/image_test/...` |

Image filename pattern: `image_{imageid}_product_{productid}.jpg`. The zip
holds 84,916 train images and 13,812 test images, exactly matching the CSV
row counts. `Y_train` aligns with `X_train` by row index (DataScientest
convention; both files share the unnamed numeric index column).

In this notebook the zip is read with `zipfile.ZipFile` and individual
images are streamed via `zf.open(name)` into PIL. The 2.5 GB archive is
not extracted to disk.

## 3. Text features

| Stat | designation (chars) | description (chars, present rows) |
|------|---------------------|-----------------------------------|
| median | 64 | 626 |
| mean   | 70 | 808 |

- `designation` is always present (0% missing). It is a short product title.
- `description` is missing in 35.1% of rows. When present it is a longer
  HTML-rich blurb (contains `<br>`, `<span>`, etc.) that must be cleaned
  before tokenisation.
- Languages on a 1000-row sample of designations (langdetect):
  French 61%, English 22%, German 7%, Catalan / Dutch / Italian /
  Romanian / Portuguese / Spanish each below 3%. Rakuten France hosts
  third-party sellers from across Europe, which explains the long tail.

## 4. Target distribution (`prdtypecode`)

- 27 distinct classes.
- Strong imbalance: max-to-min count ratio approximately 13.4x.

Top 5 classes by volume:

| prdtypecode | count |
|-------------|-------|
| 2583 | 10,209 |
| 1560 |  5,073 |
| 1300 |  5,045 |
| 2060 |  4,993 |
| 2522 |  4,989 |

Bottom 5 classes by volume:

| prdtypecode | count |
|-------------|-------|
| 60   | 832 |
| 2220 | 824 |
| 1301 | 807 |
| 1940 | 803 |
| 1180 | 764 |

## 5. Sample products per class (first three top classes)

- prdtypecode 2583: pool / outdoor accessories (e.g. liners, robotic pool cleaners).
- prdtypecode 1560: home furniture and decor.
- prdtypecode 1300: model toys and remote-control vehicles.

Concrete designations from the sampled rows are visible in section 7 of
the notebook.

## 6. Images

- Sample of 20 random training images shows uniform 500x500 RGB JPGs with
  white padding around the product. Product is centered.
- Backgrounds are clean enough that a CNN or vision transformer can be
  fine-tuned without aggressive preprocessing.
- 20 image samples (one per top-20 class) are rendered in the notebook
  by streaming directly from `images.zip`.

## 7. Key observations

1. Strong class imbalance (~13.4x). Modeling must use stratified splits,
   class-weighted loss, and report macro-F1 alongside accuracy.
2. `description` is missing for 35% of rows. The text pipeline must
   tolerate this: either fall back to designation only, or learn a
   "no-description" indicator feature.
3. `description` contains raw HTML. Strip tags before tokenisation.
4. Multilingual content. A multilingual encoder (XLM-R, mBERT, or
   multilingual fastText) is preferable to a French-only model given
   the meaningful English and German share.
5. Images are uniform and clean. A pretrained ResNet50 or ViT-B/16
   should reach a strong baseline quickly.
6. The brief mentions "over 1000 classes" referring to Rakuten's full
   internal taxonomy. The challenge file ships only 27 classes; that
   is the modeling target.

## 8. Next steps

1. Clean HTML from `description`, build a single text field
   (designation + cleaned description).
2. Build a stratified 80/10/10 train/val/test split on `prdtypecode`.
3. Text-only baseline: TF-IDF + linear SVM / logistic regression.
4. Image-only baseline: ResNet50 fine-tuned on the streamed images.
5. Multimodal head: concatenate text and image embeddings, train a
   small MLP classifier on top.
6. Track macro-F1 as the headline metric; report per-class F1 to expose
   weakness on the rare classes.
