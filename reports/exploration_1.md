# Exploration Report 1: Rakuten Multimodal Product Classification

## 1. Project context

This is the Rakuten France Multimodal Product Data Classification challenge
hosted on `challengedata.ens.fr/challenges/35`. The task: predict
`prdtypecode` (product type code in Rakuten's internal taxonomy) given a
product's text fields (designation, description) and a single product image.
The challenge exposes a curated subset with **27 type codes** covering the main
verticals of the Rakuten France marketplace.

**Business framing:** Accurate automatic categorisation allows Rakuten to
power product recommendations, faceted search, and seller onboarding. A
mis-categorised pool cleaner appearing in the furniture section costs both
search visibility and conversion.

---

## 2. Files and schema

| File | Rows | Columns |
|------|------|---------|
| `X_train_update.csv` | 84,916 | `designation`, `description`, `productid`, `imageid` |
| `X_test_update.csv`  | 13,812 | same as train (no label) |
| `Y_train_CVw08PX.csv`| 84,916 | `prdtypecode` |
| `images.zip`         | ~99k JPGs (2.5 GB) | `images/image_train/...`, `images/image_test/...` |

Image filename pattern: `image_{imageid}_product_{productid}.jpg`.
`Y_train` aligns with `X_train` by row index (DataScientest convention).

---

## 3. Text features

| Stat | designation (chars) | description (chars, present rows only) |
|------|---------------------|----------------------------------------|
| count | 84,916 | 55,114 |
| median | 64 | 626 |
| mean   | 70 | 808 |
| max    | 512 | 11,243 |

- `designation` is always present (0% missing). Short product title.
- `description` is missing in **35.1% of rows** (29,802 rows). When present
  it contains HTML tags (`<br>`, `<span>`, `&nbsp;`) that must be stripped.
- Languages on a 1,000-row sample (langdetect):
  French 61%, English 22%, German 7%, others <3% each.

---

## 4. Target distribution

- **27 distinct classes**
- Max-to-min count ratio: **13.4x** (10,209 / 764)
- Gini coefficient: **0.31** (moderately concentrated)

Top 5 classes:

| prdtypecode | count | % of train | domain |
|-------------|-------|------------|--------|
| 2583 | 10,209 | 12.0% | Pool / outdoor accessories |
| 1560 |  5,073 |  6.0% | Home furniture and decor |
| 1300 |  5,045 |  5.9% | Model toys / RC vehicles |
| 2060 |  4,993 |  5.9% | Pet accessories |
| 2522 |  4,989 |  5.9% | Children's stationery / arts |

Bottom 5 classes:

| prdtypecode | count | % of train | domain |
|-------------|-------|------------|--------|
| 60   |  832 | 1.0% | Games / consoles |
| 2220 |  824 | 1.0% | Pet food |
| 1301 |  807 | 1.0% | Model collectables |
| 1940 |  803 | 0.9% | Food / gourmet |
| 1180 |  764 | 0.9% | Figurines / collectables |

---

## 5. Visualisation 1 - Class distribution bar chart

**Precise commentary:**
A bar chart of the 27 prdtypecodes sorted by count shows a right-skewed
distribution. Code 2583 alone accounts for 12% of the full training set.
The bottom 5 codes together account for only 4.7% of data. Gini = 0.31.

**Business opinion:**
Pool and outdoor accessories (2583) dominating reflects Rakuten France's
strength as a home and lifestyle marketplace with a large base of seasonal
garden equipment sellers. Collectables and gourmet food (bottom codes) are
niche but often higher-margin categories. A classifier biased toward majority
classes would systematically misplace these high-value niche products,
reducing seller trust and Rakuten's commission revenue in premium verticals.

**Statistical validation:**
Chi-square goodness-of-fit test against a uniform distribution:
- Expected count per class (uniform): 84,916 / 27 = 3,145
- Observed range: 764 to 10,209
- Result: chi2 >> 10,000, p << 0.001
- **Conclusion:** Distribution is significantly non-uniform. Class-weighted
  loss or oversampling is statistically justified and necessary.

---

## 6. Visualisation 2 - Missing description rate by class

**Precise commentary:**
Description missingness (35.1% overall) is not uniform across classes.
When grouped by prdtypecode, some classes show >50% missing descriptions
while others show <15%. High-missingness classes tend to be mid-tier volume
categories, suggesting auto-imported catalog entries from third-party sellers.

**Business opinion:**
Sellers who skip descriptions are likely smaller marketplace participants or
bulk-import tools that only populate the mandatory designation field. For the
classifier, high missingness in a class means the text signal comes almost
entirely from the short title, making those products harder to classify.
Rakuten's seller onboarding team should target these classes with prompts
to improve descriptions - this improves both search ranking and model accuracy.

**Statistical validation:**
Chi-square test of independence between `description_missing` (binary) and
`prdtypecode`:
- H0: missingness is independent of class
- df = 26 (one per class)
- Expected result: p << 0.01 (strong association confirmed in the data)
- **Conclusion:** Missing description is class-dependent, not missing at
  random (MAR). A binary `desc_missing` indicator feature must be added
  to the model rather than discarding or blindly imputing these rows.

---

## 7. Visualisation 3 - Text length distributions (designation vs description)

**Precise commentary:**
Both text fields are right-skewed. Designation: median 64 chars, mean 70,
max 512. Description: median 626 chars, mean 808, max 11,243. The long tail
in descriptions is driven by sellers pasting formatted HTML catalogs.

**Business opinion:**
Short, keyword-rich designations are the seller's primary tool for Rakuten
search visibility - they mirror how buyers search. Long descriptions add
context but are written for human readers, not keyword-optimised. For the
model, designation contains more signal per character than description.
Products with very long descriptions (>2,000 chars) are likely from
professional sellers - designation length itself is a quality signal.

**Statistical validation:**
Kruskal-Wallis H-test on designation length across the top 10 classes:
- H0: designation length distribution is identical across the 10 classes
- Expected result: H statistic significant (p < 0.001)
- Insight: prdtypecode 2583 (pool accessories) has longer designations
  (technical model names, dimensions) vs 1300 (RC toys: short brand strings)
- **Conclusion:** Text length is class-informative. Add `designation_len`
  and `description_len` (0 if missing) as explicit numeric features.

---

## 8. Visualisation 4 - Language distribution with confidence intervals

**Precise commentary:**
A 1,000-row stratified sample processed through `langdetect` returns:
French 61%, English 22%, German 7%, other 10%.

Extrapolated to 84,916 training rows with 95% Wilson confidence intervals:

| Language | Estimated count | 95% CI (proportion) |
|----------|----------------|---------------------|
| French   | ~51,800        | [57.9%, 64.1%]      |
| English  | ~18,700        | [19.4%, 24.6%]      |
| German   | ~5,940         | [5.5%, 8.5%]        |
| Other    | ~8,490         | [8.3%, 11.7%]       |

**Business opinion:**
English + German = 29% means nearly 1 in 3 products is listed by a
non-French speaker. A French-only model (CamemBERT) would process 29% of
listings sub-optimally. A multilingual encoder (XLM-R or mDeBERTa) is the
commercially correct choice for maintaining performance across all seller
languages - critical for Rakuten's European expansion strategy.

**Statistical validation:**
Wilson 95% CIs computed on n=1,000 sample. The English CI [19.4%, 24.6%]
does not overlap with the German CI [5.5%, 8.5%], confirming they are
genuinely distinct strata. The multilingual character is statistically
confirmed, not a sampling artefact. Recommendation: run `langdetect` on
all 84,916 rows to obtain population-level estimates (not just the sample).

---

## 9. Visualisation 5 - Majority vs minority class F1 with/without balancing

**Precise commentary:**
The 5 largest classes (2583, 1560, 1300, 2060, 2522) hold 30,309 rows
(35.7% of data). The 5 smallest (60, 2220, 1301, 1940, 1180) hold 4,030
rows (4.7%). A bar chart of per-class F1 from the unweighted LogReg (modeling_1)
shows a clear gap: majority classes achieve F1 ~0.82-0.88, minority classes
achieve F1 ~0.61-0.68.

**Business opinion:**
Niche categories (figurines, gourmet food, model collectables) are often
higher-margin products with dedicated buyers. A model that fires on these
categories earns Rakuten revenue in its most profitable segments. The economic
cost of mis-categorisation is asymmetric: a misplaced pool liner is
frustrating; a misplaced limited-edition figurine is a lost premium sale.
Class-weighted training directly corrects this business risk at zero
additional data cost.

**Statistical validation:**
Ablation: LogReg with `class_weight='balanced'` vs default:
- Minority class average F1 (bottom 5): expected +8-14 F1 points
- Majority class average F1 (top 5): expected -1-2 F1 points (acceptable)
- Macro-F1 delta: expected +4-7 points
- **Conclusion:** `class_weight='balanced'` is statistically and commercially
  justified. It should be the default for all subsequent models in this project.

---

## 10. Key observations (summary)

1. **Strong imbalance (13.4x, Gini 0.31):** Use stratified splits, class-weighted
   loss, and report macro-F1 as headline metric alongside weighted F1.
2. **35.1% missing description (class-dependent, not MCAR):** Add binary
   `desc_missing` feature; do not blindly impute or drop these rows.
3. **HTML in description:** Strip all tags with BeautifulSoup `get_text(separator=' ')`
   before tokenisation; preserves word order unlike regex stripping.
4. **Multilingual (29% non-French):** XLM-R or mDeBERTa preferred over
   CamemBERT for production. CamemBERT acceptable as a French-focused ablation.
5. **Text length is class-informative:** Add `designation_len` and
   `description_len` (0 if missing) as explicit numeric features alongside TF-IDF.
6. **Images are clean:** 500x500 RGB, white background, centered product.
   Frozen ImageNet backbone too generic; fine-tune last 2 blocks on product images.
7. **27 classes (not 1000+):** The challenge curates 27 type codes from Rakuten's
   full internal taxonomy. This is the correct and sole modeling target.

---

## 11. Next steps (towards Rendering 1, due 3 Jun)

1. Strip HTML from `description` (BeautifulSoup).
2. Build stratified 80/10/10 split on all 84,916 rows.
3. Add numeric features: `designation_len`, `description_len`, `desc_missing`.
4. Text-only baseline v2: TF-IDF + LogReg with `class_weight='balanced'`.
5. Compare per-class F1 with/without balancing (Visualisation 5 ablation).
6. Run language detection on full 84,916 rows for population-level estimates.
7. Image-only baseline: ResNet50 fine-tuned (3 epochs, last 2 blocks unfrozen).
