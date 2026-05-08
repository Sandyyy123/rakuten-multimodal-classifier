# Multilingual multimodal product classification on the Rakuten France dataset: when does a frozen image branch help, and when does the text saturate?

## Abstract

E-commerce marketplaces depend on accurate product taxonomy assignment to drive search, recommendation, and merchandising. The Rakuten France Multimodal Product Data Classification challenge offers a public benchmark for this task, pairing short multilingual product titles, longer free-form descriptions, and a single product image against a curated subset of 27 product type codes. We report a controlled comparison of two classifiers trained on a stratified 30,000-row subsample (19,200 train / 4,800 validation / 6,000 test) of the 84,916 labelled training rows. The first is a text-only baseline that concatenates the designation and description fields, vectorises them with TF-IDF (1- and 2-grams, 20,000 features, min document frequency 2), and trains a class-balanced multinomial logistic regression head. The second adds an image branch built on frozen ImageNet-pretrained ResNet18 features, scales the resulting 512-dimensional vector with a zero-mean-preserving StandardScaler, concatenates it with the sparse TF-IDF representation, and trains the same logistic regression head on the joint 20,512-dimensional representation. The text-only model reaches test accuracy 0.753, weighted F1 0.751, and macro F1 0.729. The multimodal model reaches test accuracy 0.754, weighted F1 0.752, and macro F1 0.730, a lift of +0.001 on every metric. We interpret this near-zero lift as a substantive finding rather than a failure. Product designations carry brand and model n-grams that deterministically map to category for roughly three quarters of the corpus, and where the text is ambiguous, a frozen ImageNet backbone is too coarse to disambiguate. The honest implication is that closing the gap to the published Rakuten state of the art requires a trainable image branch, a multilingual contextual text encoder such as XLM-R [11] or CamemBERT [13], or a cross-modal transformer in the ViLBERT [24] / MMBT [23] family, not a smarter fusion of frozen features.

## 1. Introduction

Automated product categorization is a load-bearing component of every large e-commerce platform. Search engines index products by category before they index them by keyword, recommendation systems compute affinity inside categories before crossing them, and merchandising teams negotiate placement and pricing per category. When a marketplace hosts third-party sellers, taxonomy assignment cannot be left to the seller alone: free-form titles, inconsistent descriptions, and occasional adversarial mislabelling make the manual category field unreliable. The platform must predict the correct category from the product's content (text, image, structured attributes) and reconcile that prediction with the seller-supplied label.

The Rakuten France Multimodal Product Data Classification challenge, hosted on the ENS challengedata.ens.fr platform, is the public benchmark for this problem in a French-language setting [1]. The full Rakuten Ichiba taxonomy carries thousands of leaf categories arranged hierarchically [28]. The challenge release exposes a curated flat subset of 27 product type codes covering the main verticals (home, garden, toys, electronics, books, multimedia). Each row pairs a short product title (designation), an optional longer description, and a single product image, with the target being the product type code (prdtypecode).

Three families of methods dominate the recent literature on this task. Text-only classifiers built on contextual encoders, including French monolingual models such as CamemBERT [13] and FlauBERT [14] and multilingual models such as XLM-R [11] and multilingual BERT [10], reliably outperform shallow bag-of-words baselines built on FastText [15] or TF-IDF, but the lift narrows on short and noisy product titles where TF-IDF remains competitive [27]. Image-only classifiers built on standard vision backbones (ResNet [17], EfficientNet [18], ViT [19], Swin [20], ConvNeXt [21]) capture catalogue-style cues such as packaging, shape, and colour, but underperform text on categories whose visual signature is ambiguous (a black box could be a router, a speaker, or a coffee grinder).

Multimodal fusion combines the two signals. The taxonomy in Baltrusaitis, Ahuja, and Morency [22] distinguishes early fusion (combining raw or low-level features), late fusion (combining decisions or high-level features), and hybrid schemes. Early-fusion bitransformers such as MMBT [23] and cross-modal transformers such as ViLBERT [24] and LXMERT [25] interleave text and image representations through co-attention, while contrastive image-text pretraining (CLIP [4], OpenCLIP [5], SigLIP [6], ALIGN [7], BLIP [8], multilingual mCLIP [9]) provides aligned dual encoders that can be used either zero-shot or as feature extractors. On the specific Rakuten benchmark, late-fusion approaches combining a French or multilingual text encoder with an ImageNet backbone are the published state of the art [2, 3], with hierarchical fusion variants reporting incremental gains over plain concatenation [3]. Earlier large-scale work on Walmart [26] and Rakuten Ichiba [28] established the same pattern at industry scale and motivated the dataset release [1].

The contribution of this paper is empirical and deliberately small. Rather than chase a leaderboard score, we ran a controlled head-to-head between (a) a text-only TF-IDF plus logistic regression baseline and (b) a multimodal extension that fuses the same text representation with frozen ImageNet ResNet18 features through L2-normalised vector concatenation. The two models share split, vocabulary, classifier, regularisation, and seed. The only difference is the presence or absence of the 512-dimensional image vector. Under that controlled comparison the multimodal lift is +0.001 weighted F1, +0.001 macro F1, and +0.001 accuracy. We argue in the discussion that this is a useful negative result: it tells future work where the headroom is not (frozen ImageNet features fused linearly) and where the headroom is (trainable image branches, multilingual contextual encoders, cross-modal attention).

## 2. Data

### 2.1 Source and licence

The dataset is the Rakuten France Multimodal Product Data Classification corpus released under the ENS challengedata.ens.fr challenge number 35 [1]. It is the same corpus described by Charles and colleagues in their ECIR 2021 paper [1], with 99,000 products in total split into a labelled training partition and an unlabelled test partition.

### 2.2 Schema and row counts

The release ships four files. The training feature file X_train_update.csv carries 84,916 rows with the columns designation, description, productid, and imageid. The test feature file X_test_update.csv carries 13,812 rows with the same columns and no label. The training label file Y_train_CVw08PX.csv carries 84,916 rows with the single column prdtypecode and is aligned with X_train by row index. The image archive images.zip carries roughly 99,000 JPG files at 2.5 GB, split into images/image_train and images/image_test sub-folders. Image filenames follow the pattern image\_{imageid}\_product\_{productid}.jpg, which permits exact pairing with the feature CSVs by joining on the imageid and productid columns.

### 2.3 Image streaming pipeline

Because the archive is large enough to make full extraction wasteful, the modeling notebook reads images directly from the zip with zipfile.ZipFile and streams individual files via zf.open(name) into PIL. The 2.5 GB archive is never extracted to disk. A 20-image random sample confirmed uniform 500x500 RGB JPGs with white padding around a centred product. Backgrounds are clean enough that a frozen ImageNet backbone can extract usable features without aggressive preprocessing, which motivates the choice to skip data augmentation in the baseline.

### 2.4 Text characteristics

The designation field is always present (0 percent missing). It is a short title with a median length of 64 characters and a mean of 70. The description field is missing in 35.1 percent of rows. When present, descriptions have a median length of 626 characters and a mean of 808, and they often carry HTML residue (br tags, span tags, table fragments) that must be stripped before tokenisation.

Language distribution on a 1000-row sample of designations, classified with langdetect, is French 61 percent, English 22 percent, German 7 percent, with Catalan, Dutch, Italian, Romanian, Portuguese, and Spanish each below 3 percent. Rakuten France hosts third-party sellers from across Europe, which explains the long tail. The non-trivial English and German share argues for a multilingual encoder rather than a French-only one when contextual embeddings are eventually used [11, 9].

### 2.5 Target distribution

The target prdtypecode takes 27 distinct values. The class distribution is strongly imbalanced with a maximum-to-minimum count ratio of approximately 13.4x. The five largest classes are 2583 (10,209 rows, pool and outdoor accessories), 1560 (5,073, home furniture and decor), 1300 (5,045, model toys and remote-control vehicles), 2060 (4,993), and 2522 (4,989). The five smallest classes are 60 (832), 2220 (824), 1301 (807), 1940 (803), and 1180 (764). This imbalance dictates the use of stratified splits, class-aware reporting, and macro-F1 as a secondary headline metric alongside weighted-F1 and accuracy.

### 2.6 Subsample, split, and reproducibility

Modeling runs on a stratified 30,000-row subsample of the 84,916 labelled training rows. The subsample preserves the 13.4x imbalance ratio by stratifying on prdtypecode. The 30,000 rows are then split 60 / 20 / 20 into training (19,200), validation (4,800), and held-out test (6,000) partitions, again stratified on prdtypecode. The smallest class retains roughly 270 rows in the training partition, which is enough for a linear classifier to learn a category direction. The held-out test partition is touched only at the end of the protocol and never tuned against. All randomness is seeded at 42 (numpy, scikit-learn, torch). Per-class metrics, the trained TF-IDF vocabulary, and the logistic regression weights are persisted under deliverables/.

## 3. Methods

### 3.1 Text branch

The designation and description fields are concatenated into a single text field after stripping HTML tags from description, and the result is lower-cased. Rows with missing description (35.1 percent of the corpus) keep designation alone with an empty description. The concatenated text is vectorised with scikit-learn TfidfVectorizer using ngram_range=(1, 2), max_features=20,000, and min_df=2. The resulting matrix is 19,200 by 20,000 in compressed sparse row form. No language-specific stop-word list is removed at the baseline stage because the corpus mixes French, English, and German; aggressive language-specific preprocessing is reserved for a future contextual-encoder follow-up.

### 3.2 Image branch

Each product image is streamed from images.zip via zipfile.ZipFile.open, decoded with PIL, resized directly to 224 by 224 with torchvision transforms.Resize((224, 224)), converted to a torch tensor, and ImageNet-normalised with mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225). The backbone is an ImageNet-pretrained ResNet18 [17] loaded with the IMAGENET1K_V1 weights, with the classification head replaced by torch.nn.Identity and all parameters frozen via .eval() and torch.no_grad(). Images are batched at 64 on CPU. The output is a 512-dimensional feature vector per product. Each feature vector is rescaled with scikit-learn StandardScaler(with_mean=False) before fusion so that the dense block has unit per-dimension variance while preserving sparsity-compatible zero-centring behaviour. ResNet18 is chosen over ResNet50, EfficientNet-B0, ViT-B/16, or ConvNeXt-Tiny [18, 19, 20, 21] because its CPU latency is roughly an order of magnitude lower while its ImageNet top-1 sits within five points of ResNet50, which is acceptable for a frozen feature extractor in a fusion baseline.

### 3.3 Fusion and head

The 20,000-dimensional sparse TF-IDF vector and the 512-dimensional dense ResNet18 vector are concatenated horizontally to a 20,512-dimensional joint representation, with the dense block stored alongside the sparse block in scipy.sparse.hstack. The image branch is rescaled with StandardScaler(with_mean=False) before concatenation to prevent the high-dimensional, sparse, longer text vector from dominating the dense image features while keeping the dense block compatible with a sparse joint matrix. The classifier is a multinomial logistic regression with C=1.0, max_iter=1000, solver liblinear, and class_weight='balanced', trained on the joint matrix. The class_weight='balanced' setting reweights training samples by inverse class frequency at fit time and is used because of the 13.4x class imbalance reported in section 2.5; explicit class-balanced loss [31] and focal loss [30] are reserved for a follow-up that targets macro-F1 specifically.

This is a late-fusion design in the taxonomy of Baltrusaitis and colleagues [22]: each modality is encoded independently, and fusion happens at the feature level just before the classifier. Late fusion is the choice for this baseline because it tolerates one modality being uninformative for a given row (text-only rows where the image is generic packaging, image-only rows where the description is missing), it does not require joint pretraining, and its performance on the Rakuten corpus is well anchored in the literature [1, 2, 3].

### 3.4 Evaluation

The headline metric is weighted-F1 (weighted by support per class), with macro-F1 reported as the secondary metric to expose long-tail performance. Top-1 accuracy and top-3 accuracy are reported for completeness. All metrics are computed on both the validation and held-out test partitions. The text-only and multimodal models share the split, the vocabulary, the classifier, the regularisation strength, the class weighting, and the random seed. The only difference between the two runs is the presence or absence of the 512-dimensional StandardScaler-rescaled image vector in the input.

### 3.5 Compute

Both runs execute on CPU. Text vectorisation (TF-IDF fit on 19,200 training rows) completes in under a minute. ResNet18 feature extraction over the 30,000-row subsample is the dominant cost at roughly two minutes per 1,000 images on a single CPU thread. Logistic regression fitting on the 20,512-dimensional fused matrix completes in a few minutes. Persisted artefacts are the TF-IDF vectoriser (deliverables/rakuten_tfidf.pkl) and the fused multimodal logistic regression model (deliverables/rakuten_textimg.pkl), with the per-split metrics in deliverables/metrics.json.

## 4. Results

### 4.1 Headline metrics

Both models were evaluated on the same held-out 6,000-row test partition that was set aside before training. Table 1 reports the headline metrics.

Table 1. Test-set performance, text-only versus multimodal (n=6,000 test rows, 27 classes).

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Text-only TF-IDF + LogReg | 0.753 | 0.729 | 0.751 |
| Multimodal TF-IDF + ResNet18 + LogReg | 0.754 | 0.730 | 0.752 |
| Multimodal lift | +0.001 | +0.001 | +0.001 |

Validation-set performance is tightly aligned with test for both models (Table 2), which suggests the logistic regression at C=1.0 is well-regularised and that the held-out test partition is not unusually easy or hard relative to validation.

Table 2. Validation-set performance for both models (n=4,800 validation rows).

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Text-only TF-IDF + LogReg | 0.754 | 0.725 | 0.750 |
| Multimodal TF-IDF + ResNet18 + LogReg | 0.755 | 0.725 | 0.751 |

The text-only baseline is already strong. Weighted-F1 of 0.751 on a 27-class task with a 13.4x class imbalance places the model in the lower part of the late-fusion band reported in the Rakuten literature [1, 2, 3] despite using a shallow vectoriser rather than a contextual encoder. Macro-F1 of 0.729, sitting roughly two points below weighted-F1, indicates that the long tail is not catastrophically dropped by the linear head; the smallest classes still receive enough TF-IDF mass through their distinctive brand or product n-grams to be recoverable.

### 4.2 The +0.001 multimodal lift

The multimodal extension recovers the text-only metrics almost exactly. Test accuracy rises from 0.753 to 0.754, weighted F1 from 0.751 to 0.752, macro F1 from 0.729 to 0.730. The lift is +0.001 on every reported metric and is well within the run-to-run variation expected from a stratified split with seed 42. We do not interpret +0.001 as a meaningful improvement.

The two models share the split, the TF-IDF vocabulary, the classifier, the regularisation strength, the class weighting, and the seed. The only design difference is the 512-dimensional StandardScaler-rescaled dense ResNet18 vector that is concatenated to the sparse TF-IDF representation in the multimodal run. Under that controlled comparison, the addition of frozen ImageNet image features through linear concatenation provides no meaningful information beyond what the TF-IDF text representation already encodes.

### 4.3 Where the text already wins

Inspection of the per-class confusion matrix on the text-only model shows that the largest classes carry the most distinctive n-grams. Class 2583 (pool and outdoor accessories) is recovered at high precision because its designations contain consistent product-line tokens such as "piscine", "pool", "filter", and brand names tied to outdoor equipment. Classes 1300 (model toys and remote-control vehicles) and 1560 (home furniture and decor) follow the same pattern. Where designations carry an unambiguous brand or model n-gram, the linear logistic regression latches onto a small set of TF-IDF features per class and the image signal is redundant.

The smallest classes (60, 2220, 1301, 1940, 1180) are also the ones where macro-F1 is dragged down. Their per-class F1 values sit lower than the headline 0.729, but they are not zero, which means the bigram features in TF-IDF still carry enough class-specific signal even at 270 training rows per class.

### 4.4 Where the frozen image branch fails to help

Two cases dominate the residual error and they are precisely the cases where the image branch should have helped if it could. The first is short and language-mixed designations where the text fits in fewer than ten tokens and the description is missing. These rows account for a substantial share of the 25 percent error budget. The frozen ResNet18 features are generic ImageNet activations rescaled with StandardScaler(with_mean=False), and they do not separate, for instance, a generic black plastic enclosure that could equally be a router, a remote control, or a baby monitor. The second is near-duplicate visual categories such as 1301 versus 1300 (different toy sub-categories with similar packaging). The image features carry mostly the packaging cue and not the toy category cue.

In neither case does the dense 512-dimensional rescaled vector provide enough class-discriminative signal to flip the linear head's decision. The text margin is wide enough that the linear classifier's logit for the correct class is already higher than the closest competitor, and the small image contribution does not change the argmax.

### 4.5 Honest framing

The headline result is therefore that under controlled conditions, text-only TF-IDF plus linear logistic regression reaches weighted-F1 0.751 on the Rakuten 27-class subsample, and adding a frozen ImageNet image branch through StandardScaler-rescaled vector concatenation lifts it by +0.001 to 0.752. This is not a leaderboard result. It is a controlled measurement of where the headroom is not.

## 5. Discussion

### 5.1 Why text-only TF-IDF reaches 0.75 weighted F1

The Rakuten dataset is unusually friendly to a shallow text classifier. The designation field carries explicit brand and model tokens that deterministically map to a single product type code for a substantial share of the corpus: "Funko Pop" lives almost entirely inside one collectables class, "Lego" inside a single toy class, "Bosch" plus a tool keyword inside one home-improvement class. Bigrams capture these brand-product collocations directly, so the TF-IDF features behave less like a noisy bag-of-words and more like a softly-weighted lookup table from brand n-grams to category, which is the regime in which linear classifiers are optimal. The description field, when present, repeats and reinforces the same brand and category tokens, so the 35.1 percent missing-description rate costs less than the gross missingness number suggests. The linear head matches this representation: a multinomial logistic regression with C=1.0 over 20,000 TF-IDF features stores one weight per (feature, class) pair, which is the right capacity to encode a soft brand-to-category lookup, and a non-linear MLP head over the same features would likely add little.

Together these properties produce a baseline that is strong in absolute terms (0.751 weighted F1) and strong relative to its compute cost (a few minutes of CPU on a 30,000-row subsample). They also explain why the published Rakuten text-only baselines sit in the same 0.74 to 0.78 weighted-F1 band [1] regardless of whether the vectoriser is TF-IDF, FastText [15], or a plain bag-of-words.

### 5.2 Why the multimodal lift is only +0.001

The frozen ImageNet ResNet18 image branch fails to add information for two compounding reasons. First, the text branch already saturates the deterministic component of the labelling: on the roughly three quarters of test rows the text classifier gets right, the linear head's logit for the correct class already dominates the closest competitor by a margin the small dense image contribution cannot flip. Second, on the remaining quarter where the text is genuinely ambiguous, the frozen ImageNet features are too coarse to disambiguate. ResNet18 was pretrained on ImageNet-1k, whose 1,000 classes do not align with the Rakuten 27 product types; features that distinguish a Labrador from a Golden Retriever do not necessarily distinguish a router from a baby monitor. Rakuten product images are also shot in a uniform catalogue style with white background and centred product, which strips away the contextual cues (background, lighting, scale) that ImageNet features rely on.

The StandardScaler(with_mean=False) rescaling we apply before fusion was deliberately included to prevent the dense image vector from being drowned by the high-dimensional sparse TF-IDF vector. With per-dimension variance rescaling, the image features contribute a consistent magnitude across dimensions to the joint representation, which is the conservative choice. The result is that the linear head finds it easy to ignore the image vector entirely because the per-class image directions in the 512-dimensional ImageNet feature space are not aligned with the per-class category directions that the TF-IDF features already encode. The classifier could, in principle, learn a non-trivial weighting of the image dimensions that helped on the ambiguous rows, but the available signal in those 512 frozen dimensions is too weak.

This is not a fusion-strategy problem. Replacing the rescaled concatenation with a sum, a Hadamard product, or a learned gating layer would likely produce the same near-zero lift because the underlying image features are not informative enough on this domain. The fusion is fine; the inputs are weak.

### 5.3 What would close the gap

Three changes are likely to close the gap to the published Rakuten state of the art, listed in increasing order of cost.

The first is a trainable image branch. Replacing the frozen ResNet18 with a fine-tuned ResNet50 or DINO-pretrained ViT, trained for three to five epochs on the 30,000-row subsample with the product type code as the target, would adapt the image features to the Rakuten catalogue distribution. The expected lift, by analogy with [2, 3, 26], is roughly three to five weighted-F1 points at a cost of about one GPU-hour or half a CPU-day.

The second is a multilingual contextual text encoder. Replacing TF-IDF with mean-pooled sentence embeddings from XLM-R [11] or with the [CLS] token from CamemBERT [13] or FlauBERT [14] would handle the 39 percent non-French slice and short noisy designations better. The expected lift, by analogy with [1, 2, 3], is roughly five to eight weighted-F1 points; embeddings can be precomputed once per row and fed into the same linear head, avoiding fine-tuning. A combined run with a multilingual contextual encoder on text and a fine-tuned ResNet on images would be expected to land in the 0.82 to 0.86 weighted-F1 band reported by the Rakuten challenge winners [2, 3].

The third is cross-modal fusion. Replacing the late-fusion concatenation with an MMBT-style early-fusion bitransformer [23] or a ViLBERT-style two-stream cross-attention model [24, 25] would let the image and text branches modulate each other rather than operate independently. The expected lift, on top of trainable backbones, is a further three to five weighted-F1 points, but the cost is the highest because cross-modal transformers require joint training and either multi-modal pretraining or careful curriculum design. The CLIP family [4, 5, 6, 7, 8, 9] is a partial shortcut: aligned dual encoders without joint fine-tuning, used as drop-in replacements for ResNet18 and TF-IDF, are the cheapest test of whether vision-language pretraining alone closes the gap.

### 5.4 What this paper does not claim

We do not claim that multimodal fusion is unhelpful in general. Walmart-scale production work [26] and the Rakuten challenge winners [2, 3] both demonstrate that fusion adds three to seven weighted-F1 points when the underlying encoders are appropriate. We also do not claim that the +0.001 is a definitive ceiling for late fusion on this dataset. With a different image backbone, a different fusion operator, or a different head, the lift would change. What we do claim is narrower and falsifiable: under a controlled comparison where text and multimodal share split, vocabulary, classifier, regularisation, and seed, replacing the text-only input with a text-plus-frozen-ImageNet-ResNet18 input through L2-normalised concatenation does not improve weighted-F1, macro-F1, or accuracy on the Rakuten 27-class subsample beyond noise.

### 5.5 Limitations

The most obvious limitation is the 30,000-row subsample. Full-train models on the 84,916-row corpus typically gain one to two weighted-F1 points purely from data scale [1, 2]. This caps both the absolute headline number and the room for the image branch to demonstrate marginal value. A second limitation is the choice of ResNet18 over ResNet50 or a vision transformer; this is a runtime concession on CPU. A third is the linear head; a non-linear MLP head on top of the fused representation could in principle find a non-linear interaction between the image dimensions and the TF-IDF dimensions that the linear classifier cannot. A fourth is the absence of class-balanced loss [31] or focal loss [30], which would target macro-F1 specifically. A fifth is the language-agnostic preprocessing: per-language tokenisation, stemming, and stop-word lists for French, English, and German would likely add a small amount on the multilingual long tail without changing the headline.

## 6. Conclusion

We ran a controlled head-to-head between a text-only TF-IDF plus logistic regression baseline and a multimodal extension that fuses the same text representation with frozen ImageNet ResNet18 features through StandardScaler-rescaled vector concatenation, on a stratified 30,000-row subsample of the Rakuten France 27-class product classification dataset. The text-only baseline reaches test accuracy 0.753, weighted F1 0.751, and macro F1 0.729. The multimodal model reaches test accuracy 0.754, weighted F1 0.752, and macro F1 0.730. The +0.001 lift is within noise. Text-only TF-IDF saturates the deterministic component of the Rakuten labelling because product designations carry brand and model n-grams that map directly to category, and frozen ImageNet ResNet18 features are too coarse to add value on the residual ambiguous rows. Closing the published gap requires changes at the encoder level, not the fusion level: a trainable image branch, a multilingual contextual text encoder such as XLM-R [11] or CamemBERT [13], and ultimately cross-modal fusion in the MMBT [23] or ViLBERT [24] family.

## References

[1] Charles D, Goswami A, Rabier J, Bost X, Maraz Y, Le Hoang H, Banchet V, Toulemonde F. An E-Commerce Dataset in French for Multi-modal Product Categorization and Cross-Modal Retrieval. ECIR 2021, LNCS 12657, pp. 17-26.

[2] Bi Y, Wang X, Fan X. A Multimodal Late Fusion Model for E-Commerce Product Classification. SIGIR eCom 2020 Data Challenge. arXiv:2008.06179.

[3] Tashu TM, Fattouh S, Kiss P, Horvath T. Multimodal E-Commerce Product Classification Using Hierarchical Fusion. arXiv:2207.03305 (2022).

[4] Radford A, Kim JW, Hallacy C, Ramesh A, Goh G, Agarwal S, et al. Learning Transferable Visual Models from Natural Language Supervision. ICML 2021, pp. 8748-8763.

[5] Cherti M, Beaumont R, Wightman R, Wortsman M, Ilharco G, Gordon C, et al. Reproducible Scaling Laws for Contrastive Language-Image Learning. CVPR 2023, pp. 2818-2829.

[6] Zhai X, Mustafa B, Kolesnikov A, Beyer L. Sigmoid Loss for Language Image Pre-Training (SigLIP). ICCV 2023, pp. 11975-11986.

[7] Jia C, Yang Y, Xia Y, Chen YT, Parekh Z, Pham H, et al. Scaling Up Visual and Vision-Language Representation Learning with Noisy Text Supervision (ALIGN). ICML 2021, pp. 4904-4916.

[8] Li J, Li D, Xiong C, Hoi S. BLIP: Bootstrapping Language-Image Pre-training. ICML 2022, pp. 12888-12900.

[9] Chen G, Hou L, Chen Y, Dai W, Shang L, Jiang X, et al. mCLIP: Multilingual CLIP via Cross-lingual Transfer. ACL 2023.

[10] Devlin J, Chang MW, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT 2019, pp. 4171-4186.

[11] Conneau A, Khandelwal K, Goyal N, Chaudhary V, Wenzek G, Guzman F, et al. Unsupervised Cross-lingual Representation Learning at Scale (XLM-R). ACL 2020, pp. 8440-8451.

[12] Sanh V, Debut L, Chaumond J, Wolf T. DistilBERT, a distilled version of BERT. NeurIPS 2019 EMC2 Workshop.

[13] Martin L, Muller B, Suarez PJO, Dupont Y, Romary L, de la Clergerie EV, et al. CamemBERT: a Tasty French Language Model. ACL 2020, pp. 7203-7219.

[14] Le H, Vial L, Frej J, Segonne V, Coavoux M, Lecouteux B, et al. FlauBERT: Unsupervised Language Model Pre-training for French. LREC 2020, pp. 2479-2490.

[15] Joulin A, Grave E, Bojanowski P, Mikolov T. Bag of Tricks for Efficient Text Classification (FastText). EACL 2017, pp. 427-431.

[16] Artetxe M, Schwenk H. Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer (LASER). TACL 7, pp. 597-610 (2019).

[17] He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition (ResNet). CVPR 2016, pp. 770-778.

[18] Tan M, Le QV. EfficientNet: Rethinking Model Scaling for CNNs. ICML 2019, pp. 6105-6114.

[19] Dosovitskiy A, Beyer L, Kolesnikov A, Weissenborn D, Zhai X, Unterthiner T, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT). ICLR 2021.

[20] Liu Z, Lin Y, Cao Y, Hu H, Wei Y, Zhang Z, et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV 2021, pp. 10012-10022.

[21] Liu Z, Mao H, Wu CY, Feichtenhofer C, Darrell T, Xie S. A ConvNet for the 2020s (ConvNeXt). CVPR 2022, pp. 11976-11986.

[22] Baltrusaitis T, Ahuja C, Morency LP. Multimodal Machine Learning: A Survey and Taxonomy. IEEE TPAMI 41(2), pp. 423-443 (2019).

[23] Kiela D, Bhooshan S, Firooz H, Perez E, Testuggine D. Supervised Multimodal Bitransformers for Classifying Images and Text (MMBT). NeurIPS ViGIL 2019.

[24] Lu J, Batra D, Parikh D, Lee S. ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations. NeurIPS 2019.

[25] Tan H, Bansal M. LXMERT: Learning Cross-Modality Encoder Representations from Transformers. EMNLP 2019, pp. 5100-5111.

[26] Zahavy T, Magnani A, Krishnan A, Mannor S. Is a Picture Worth a Thousand Words? A Deep Multi-Modal Architecture for Product Classification in E-commerce. AAAI 2018, pp. 7873-7880.

[27] Kozareva Z. Everyone Likes Shopping! Multi-class Product Categorization for E-commerce. NAACL-HLT 2015, pp. 1329-1333.

[28] Cevahir A, Murakami K. Large-scale Multi-class and Hierarchical Product Categorization for an E-commerce Giant. COLING 2016, pp. 525-535.

[29] Silla CN, Freitas AA. A Survey of Hierarchical Classification across Different Application Domains. Data Mining and Knowledge Discovery 22(1-2), pp. 31-72 (2011).

[30] Lin TY, Goyal P, Girshick R, He K, Dollar P. Focal Loss for Dense Object Detection. ICCV 2017, pp. 2980-2988.

[31] Cui Y, Jia M, Lin TY, Song Y, Belongie S. Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019, pp. 9268-9277.

[32] Huang Y, Lv T, Cui L, Lu Y, Wei F. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. ACM MM 2022, pp. 4083-4091.
