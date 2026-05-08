# VALIDATOR Report - Project 06 Rakuten Multimodal

## Summary

Overall status: **PASS-WITH-WARNINGS**

The project artefacts are structurally sound: both notebooks parse as valid JSON, the manuscript hits the 4000-5000 word target (5183 words, slightly over), the presentation HTML is fully self-contained (zero external resources), all 32 references are cited and 5/5 spot-checked DOIs resolve live on CrossRef, no em-dash characters appear in any text artefact, and no AI-tell phrases were found. The IMRaD structure is complete (Title, Abstract, Introduction, Data, Methods, Results, Discussion, Conclusion, References). Two warnings: (1) clear method drift between manuscript narrative and the modeling notebook on five hyperparameter / preprocessing choices (solver, max_iter, min_df, image normalisation strategy, image resize+crop pipeline), and (2) checkpoint.json is absent and src/ contains no Python module (model code lives entirely in notebooks/build_modeling_nb.py + 03_modeling.ipynb, which is allowed for projects #1-#8).

---

## Findings (one per line)

### 1. Notebook validity
- [PASS] notebooks/01_eda.ipynb parses as JSON (22 cells)
- [PASS] notebooks/03_modeling.ipynb parses as JSON (28 cells)

### 2. Python script syntax (adapted - src/ empty, model code in notebooks/)
- [PASS] notebooks/build_eda_nb.py - ast.parse OK
- [PASS] notebooks/build_modeling_nb.py - ast.parse OK
- [WARN] src/ is empty; per QA spec, projects #1-#8 may have model code in notebooks/, which is the case here. Not a FAIL.

### 3. Manuscript word count
- [WARN] manuscripts/manuscript.md word count = 5183 (target 4000-5000). Overshoot by 183 words (~3.7%); minor.

### 4. Self-contained HTML
- [PASS] deliverables/presentation.html has 0 external `href="http"` or `src="http"` resources. Fully inline.

### 5. IMRaD completeness
- [PASS] Title (line 1)
- [PASS] Abstract (sec)
- [PASS] Introduction (sec 1)
- [PASS] Data (sec 2; replaces or supplements Methods data subsection)
- [PASS] Methods (sec 3)
- [PASS] Results (sec 4)
- [PASS] Discussion (sec 5)
- [PASS] Conclusion (sec 6)
- [PASS] References (sec at end)

### 6. Method drift (manuscript vs modeling notebook)
Methods named in manuscript Methods sec 3.1-3.5:
- TF-IDF (TfidfVectorizer, ngram (1,2), max_features 20000, min_df=3) - [FAIL] notebook uses `min_df=2` (manuscript says 3)
- Logistic regression (C=1.0, max_iter=300, solver=lbfgs, no class weighting) - [FAIL] notebook uses `solver='liblinear'`, `max_iter=1000`, `class_weight='balanced'` (manuscript says no weighting at baseline). Three drifts in one sentence.
- ResNet18 ImageNet1K_V1 frozen, head -> Identity, .eval() + no_grad - [PASS] all in notebook
- Image preprocessing: shorter side resized to 256, centre-cropped to 224, ImageNet-normalised - [FAIL] notebook uses direct `transforms.Resize((224, 224))` (no centre-crop, no shorter-side-256 step)
- L2-normalisation of image features before fusion - [FAIL] notebook uses `StandardScaler(with_mean=False)` instead; not L2 unit-norm
- scipy.sparse.hstack concat of TF-IDF (sparse) and image features (dense) - [PASS] in notebook
- Per-modality late fusion at feature level - [PASS]
- Weighted-F1 + macro-F1 + top-1 accuracy + top-3 accuracy - [PASS] (manuscript Methods 3.4 lists three; notebook also reports top-3, which is a manuscript omission, not drift)

Net: 4 method-drift issues in section 3 (min_df, solver, max_iter, class_weight, image normalisation pipeline, L2 vs StandardScaler). The manuscript Methods narrative does NOT match the executed notebook configuration. This is a reproducibility gap.

### 7. Citation drift
- [PASS] Manuscript uses numeric citations [1] through [32]
- [PASS] All 32 unique citation numbers (1, 2, 3, ... 32) appear in manuscripts/references.md
- [PASS] No orphan citation found

### 8. Re-verify 5 random references via CrossRef live
Picked: refs 1 (Charles 2021 Rakuten ECIR), 9 (mCLIP 2023), 22 (Baltrusaitis 2019), 27 (Kozareva 2015), 29 (Silla Freitas 2011).
- [PASS] 10.1007/978-3-030-72113-8_2 - HTTP 200, title matches "An E-Commerce Dataset in French for Multi-modal Product Categorization..."
- [PASS] 10.18653/v1/2023.acl-long.728 - HTTP 200, title matches "mCLIP: Multilingual CLIP via Cross-lingual Transfer"
- [PASS] 10.3115/v1/N15-1147 - HTTP 200, title matches "Everyone Likes Shopping! Multi-class Product Categorization..."
- [PASS] 10.1007/s10618-010-0175-9 - HTTP 200, title matches "A survey of hierarchical classification across different application domains"
- [PASS] 10.1109/TPAMI.2018.2798607 - HTTP 200, title matches "Multimodal Machine Learning: A Survey and Taxonomy" (this DOI was inferred from the journal record; references.md cites only arXiv 1705.09406, which is consistent with the same paper)

### 9. Em-dash scan
- [PASS] manuscripts/references.md = 0
- [PASS] manuscripts/manuscript.md = 0
- [PASS] notebooks/01_eda.ipynb = 0
- [PASS] notebooks/03_modeling.ipynb = 0
- [PASS] deliverables/presentation.html = 0
- (brief.pdf skipped - binary)

### 10. AI-tell scan
- [PASS] grep -riE 'verified by [0-9]+ agents|AI-verified|cross-checked by Claude' returned 0 hits across the project folder.

### 11. Checkpoint schema
- [WARN] checkpoint.json is ABSENT from project root. Cannot verify required keys (project_number, title, methodology, status). Recommend adding checkpoint.json for portfolio consistency.

### Bonus: deliverable artefacts (project #1-#8 was executed)
- [PASS] deliverables/rakuten_textimg.pkl present (4.43 MB) - fused multimodal model
- [PASS] deliverables/rakuten_tfidf.pkl present (786 KB) - TF-IDF vectoriser
- [PASS] deliverables/metrics.json present (1 KB)
- [PASS] deliverables/presentation.html present (29.5 KB, self-contained)
- [PASS] data/ contains the four required Rakuten files (X_train, X_test, Y_train, images.zip 2.56 GB)

---

## Blockers

None. All checks executed. CrossRef API responded for all 5 picked DOIs.
