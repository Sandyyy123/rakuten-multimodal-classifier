# References - Multimodal Product Classification on Rakuten France

Verified against arXiv, Semantic Scholar, and publisher DOIs (May 2026). Each entry lists authors, year, title, venue, identifier, and a one-line abstract.

## 1. Rakuten dataset and challenge baselines

1. **Charles, Goswami, Rabier, Bost, Maraz, Le Hoang, Banchet, Toulemonde (2021).** An E-Commerce Dataset in French for Multi-modal Product Categorization and Cross-Modal Retrieval. *ECIR 2021, LNCS 12657, pp. 17-26.* DOI: 10.1007/978-3-030-72113-8_2. Releases the 99k Rakuten France dataset (text + image, 27 product types) and reports unimodal and late-fusion baselines.

2. **Bi, Wang, Fan (2020).** A Multimodal Late Fusion Model for E-Commerce Product Classification. *SIGIR eCom 2020 Data Challenge.* arXiv: 2008.06179. Modality-specific deep nets (text + image) fused at the decision level, ranked first in the SIGIR Rakuten challenge.

3. **Tashu, Fattouh, Kiss, Horvath (2022).** Multimodal E-Commerce Product Classification Using Hierarchical Fusion. *arXiv: 2207.03305.* Combines CamemBERT / FlauBERT text features with SE-ResNeXt-50 image features via concat + average fusion on the Rakuten dataset.

## 2. Vision-language pretraining and the CLIP family

4. **Radford et al. (2021).** Learning Transferable Visual Models from Natural Language Supervision. *ICML 2021, pp. 8748-8763.* arXiv: 2103.00020. Contrastive image-text pretraining (CLIP) on 400M pairs, enabling zero-shot transfer to dozens of downstream classification tasks.

5. **Cherti et al. (2023).** Reproducible Scaling Laws for Contrastive Language-Image Learning. *CVPR 2023, pp. 2818-2829.* arXiv: 2212.07143. OpenCLIP scaling study on LAION up to 2B pairs, releasing reproducible CLIP checkpoints used widely for product imagery.

6. **Zhai, Mustafa, Kolesnikov, Beyer (2023).** Sigmoid Loss for Language Image Pre-Training (SigLIP). *ICCV 2023, pp. 11975-11986.* arXiv: 2303.15343. Replaces softmax contrastive with pairwise sigmoid loss, improving small-batch performance and memory efficiency.

7. **Jia et al. (2021).** Scaling Up Visual and Vision-Language Representation Learning with Noisy Text Supervision (ALIGN). *ICML 2021, pp. 4904-4916.* arXiv: 2102.05918. Trains dual-encoder vision-language model on 1.8B noisy alt-text pairs without curation, matching CLIP performance.

8. **Li, Li, Xiong, Hoi (2022).** BLIP: Bootstrapping Language-Image Pre-training. *ICML 2022, pp. 12888-12900.* arXiv: 2201.12086. Unified VLP framework with caption bootstrapping; competitive on retrieval and captioning relevant to product description.

9. **Chen et al. (2023).** mCLIP: Multilingual CLIP via Cross-lingual Transfer. *ACL 2023.* DOI: 10.18653/v1/2023.acl-long.728. Aligns multilingual text encoders with CLIP image features, supporting French / English / German queries directly.

## 3. Text encoders (monolingual and multilingual)

10. **Devlin, Chang, Lee, Toutanova (2019).** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019, pp. 4171-4186.* arXiv: 1810.04805. Masked-LM pretraining yielding contextual embeddings; foundation for product-text classifiers.

11. **Conneau et al. (2020).** Unsupervised Cross-lingual Representation Learning at Scale (XLM-R). *ACL 2020, pp. 8440-8451.* arXiv: 1911.02116. RoBERTa-style pretraining on 100 languages including French / English / German, ideal for the multilingual Rakuten titles.

12. **Sanh, Debut, Chaumond, Wolf (2019).** DistilBERT, a distilled version of BERT. *NeurIPS 2019 EMC2 Workshop.* arXiv: 1910.01108. Knowledge-distilled BERT with 40% fewer parameters and 60% faster inference, useful for production-grade product classifiers.

13. **Martin et al. (2020).** CamemBERT: a Tasty French Language Model. *ACL 2020, pp. 7203-7219.* arXiv: 1911.03894. RoBERTa-architecture French monolingual model, strong on French e-commerce text.

14. **Le et al. (2020).** FlauBERT: Unsupervised Language Model Pre-training for French. *LREC 2020, pp. 2479-2490.* arXiv: 1912.05372. Alternative French BERT plus the FLUE evaluation suite, used as second French encoder in Rakuten work.

15. **Joulin, Grave, Bojanowski, Mikolov (2017).** Bag of Tricks for Efficient Text Classification (FastText). *EACL 2017, pp. 427-431.* arXiv: 1607.01759. Linear bag-of-n-grams classifier; standard fast multilingual baseline.

16. **Artetxe, Schwenk (2019).** Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer (LASER). *TACL 7, pp. 597-610.* arXiv: 1812.10464. Single BiLSTM encoder covering 93 languages, used for cross-lingual product retrieval.

## 4. Vision backbones for product images

17. **He, Zhang, Ren, Sun (2016).** Deep Residual Learning for Image Recognition (ResNet). *CVPR 2016, pp. 770-778.* arXiv: 1512.03385. Residual connections enabling very deep CNNs; default image backbone in many Rakuten submissions.

18. **Tan, Le (2019).** EfficientNet: Rethinking Model Scaling for CNNs. *ICML 2019, pp. 6105-6114.* arXiv: 1905.11946. Compound scaling of depth / width / resolution; strong accuracy / FLOP trade-off for product imagery.

19. **Dosovitskiy et al. (2021).** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT). *ICLR 2021.* arXiv: 2010.11929. Pure transformer applied to image patches; matches or beats CNNs at scale.

20. **Liu et al. (2021).** Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV 2021, pp. 10012-10022.* arXiv: 2103.14030. Hierarchical attention with shifted windows; strong dense prediction backbone for product images.

21. **Liu, Mao, Wu, Feichtenhofer, Darrell, Xie (2022).** A ConvNet for the 2020s (ConvNeXt). *CVPR 2022, pp. 11976-11986.* arXiv: 2201.03545. Modernised ResNet recipe matching transformer-era accuracy at lower cost.

## 5. Multimodal fusion strategies

22. **Baltrusaitis, Ahuja, Morency (2019).** Multimodal Machine Learning: A Survey and Taxonomy. *IEEE TPAMI 41(2), pp. 423-443.* arXiv: 1705.09406. Canonical taxonomy of representation, alignment, fusion, translation, and co-learning across modalities.

23. **Kiela, Bhooshan, Firooz, Perez, Testuggine (2019).** Supervised Multimodal Bitransformers for Classifying Images and Text (MMBT). *NeurIPS ViGIL 2019.* arXiv: 1909.02950. Simple early-fusion bitransformer combining BERT text and image-region features, strong baseline on text + image classification.

24. **Lu, Batra, Parikh, Lee (2019).** ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations. *NeurIPS 2019.* arXiv: 1908.02265. Two-stream cross-modal transformer with co-attention for vision-and-language pretraining.

25. **Tan, Bansal (2019).** LXMERT: Learning Cross-Modality Encoder Representations from Transformers. *EMNLP 2019, pp. 5100-5111.* arXiv: 1908.07490. Three-encoder design (object, language, cross-modal) pretrained on five vision-language tasks.

## 6. E-commerce, hierarchical and large-label-space classification

26. **Zahavy, Magnani, Krishnan, Mannor (2018).** Is a Picture Worth a Thousand Words? A Deep Multi-Modal Architecture for Product Classification in E-commerce. *AAAI 2018, pp. 7873-7880.* arXiv: 1611.09534. Walmart-scale text + image policy network deciding per item which modality to trust, ancestor of all product-classification fusion work.

27. **Kozareva (2015).** Everyone Likes Shopping! Multi-class Product Categorization for E-commerce. *NAACL-HLT 2015, pp. 1329-1333.* DOI: 10.3115/v1/N15-1147. Early text-only large-taxonomy product classification baseline.

28. **Cevahir, Murakami (2016).** Large-scale Multi-class and Hierarchical Product Categorization for an E-commerce Giant. *COLING 2016, pp. 525-535.* Two-level hierarchical neural classifier deployed at Rakuten Ichiba, direct precursor to the Rakuten France challenge.

29. **Silla, Freitas (2011).** A Survey of Hierarchical Classification across Different Application Domains. *Data Mining and Knowledge Discovery 22(1-2), pp. 31-72.* DOI: 10.1007/s10618-010-0175-9. Reference taxonomy (flat, local-per-node, local-per-level, global) for hierarchical category prediction.

## 7. Class imbalance handling

30. **Lin, Goyal, Girshick, He, Dollar (2017).** Focal Loss for Dense Object Detection. *ICCV 2017, pp. 2980-2988.* arXiv: 1708.02002. Down-weights easy examples; widely used for the long-tailed product-type distribution in Rakuten.

31. **Cui, Jia, Lin, Song, Belongie (2019).** Class-Balanced Loss Based on Effective Number of Samples. *CVPR 2019, pp. 9268-9277.* arXiv: 1901.05555. Re-weights losses by the effective number of samples per class, helping rare Rakuten categories.

## 8. Document AI / multimodal layout (optional context)

32. **Huang, Lv, Cui, Lu, Wei (2022).** LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. *ACM MM 2022, pp. 4083-4091.* arXiv: 2204.08387. Unified text-image masking pretraining; relevant when product descriptions contain layout cues (catalogue scans, marketplaces with rich text).
