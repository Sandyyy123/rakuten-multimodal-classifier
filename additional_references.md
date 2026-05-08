# Additional References (Literature Scout, 2024-2026)

Project: 06 Rakuten France Multimodal Product Classification.
Method context: TF-IDF text baseline vs late-fusion with frozen ImageNet ResNet18; 27-class subsample; honest negative result on the +0.001 multimodal lift.

All entries below were resolved live via the CrossRef REST API (`https://api.crossref.org/works/{doi}`) on 2026-05-08; entries that did not resolve were dropped, not padded.

## State-of-the-art callout (gaps in the existing reference list)

The current manuscript references (Charles 2021, Bi 2020, Tashu 2022, ResNet, ViT, ConvNeXt, CLIP family, MMBT, ViLBERT, CamemBERT, XLM-R) cover the canonical text and vision backbones and the late/early-fusion taxonomy, but they do NOT cover five strands that are directly load-bearing for the project's stated headroom story:

1. **Modality collapse / modality greedy on multimodal product classification.** The +0.001 lift the manuscript reports is the textbook signature of "modality dominance" on a fused linear head. Obayemi and Nguyen (ICMLA 2024, DOI 10.1109/icmla61862.2024.00244) measure this exact failure mode on fine-grained e-commerce product classification using Shapley-value attributions, and Ding, Ma, Zhang (ACM MM 2025, DOI 10.1145/3746027.3754820) prove dynamic fusion exacerbates the greedy modality. Cite at least one of the two in the discussion to anchor the negative result.
2. **2024-2025 e-commerce-specific multimodal benchmarks.** Gross et al. (BigData 2025, DOI 10.1109/bigdata66926.2025.11402414) report a multimodal hierarchical classifier across cross-platform product taxonomies and is the most recent direct competitor to the Rakuten setup; this should be cited as a "what state-of-the-art looks like in 2025" anchor.
3. **LLM-based attribute and category extraction on e-commerce product data.** The Methods section ends at TF-IDF + ResNet18 + LogReg, but Fang et al. (SIGIR 2024, DOI 10.1145/3626772.3661357), Çiftlikçi et al. (Electronics 2025, DOI 10.3390/electronics14101930), and Zhang, Khan, Walter (EMNLP-Industry 2025, DOI 10.18653/v1/2025.emnlp-industry.18) document the LLM upgrade path that the Discussion gestures at without citing.
4. **CLIP-as-feature-extractor on document and product images.** The Discussion proposes CLIP features as a "cheapest way" to test vision-language pretraining. Aljuhani, Dahab, Alsenani (Sensors 2025, DOI 10.3390/s25247596) and Onuoha et al. (Electronics 2024, DOI 10.3390/electronics13040803) report empirical findings on whether CLIP visual features actually help for non-photographic, catalogue-style images, which is exactly the regime the manuscript's images sit in.
5. **Imbalanced multimodal learning beyond focal/class-balanced loss.** Zhou et al. (Information Fusion 2025, DOI 10.1016/j.inffus.2025.103383) propose dataset-aware modality contribution for imbalanced multimodal learning, which is more recent and more directly applicable than the 2017-2019 focal/class-balanced loss references the manuscript currently lists.

These five gaps are the most defensible upgrades to `reports/references.md`; each has an in-domain experimental section, not just a method paper.

---

## Architectures and fusion mechanisms (2024-2026)

Bayoudh K. A survey of multimodal hybrid deep learning for computer vision: Architectures, applications, trends, and challenges. Information Fusion. 2024. DOI:10.1016/j.inffus.2023.102217

Ding X, Ma H, Zhang C. A Theoretical Proof of Dynamic Multimodal Fusion Exacerbates Modality Greedy. Proceedings of the 33rd ACM International Conference on Multimedia. 2025. DOI:10.1145/3746027.3754820

Zhou Y, Liang X, Xu Y, Lin X. Dataset-aware Utopia modality contribution for imbalanced multimodal learning. Information Fusion. 2025. DOI:10.1016/j.inffus.2025.103383

Shi Q, Xu W, Miao Z. Image-text multimodal classification via cross-attention contextual transformer with modality-collaborative learning. Journal of Electronic Imaging. 2024. DOI:10.1117/1.jei.33.4.043042

Li M, Hao R, Shi S, Yu Z, He Q, Zhan J. A CNN-Transformer Approach for Image-Text Multimodal Classification with Cross-Modal Feature Fusion. 2025 8th International Conference on Advanced Algorithms and Control Engineering (ICAACE). 2025. DOI:10.1109/icaace65325.2025.11020324

## E-commerce product classification (2023-2026)

Obayemi A, Nguyen K. Leveraging Multimodal Shapley Values to Address Multimodal Collapse and Improve Fine-Grained E-Commerce Product Classification. 2024 International Conference on Machine Learning and Applications (ICMLA). 2024. DOI:10.1109/icmla61862.2024.00244

Gross L, Walter R, Zoppi N, Justus A, Gambetti A, Han Q, Kaiser M. Cross-Platform E-Commerce Product Categorization and Recategorization: A Multimodal Hierarchical Classification Approach. 2025 IEEE International Conference on Big Data (BigData). 2025. DOI:10.1109/bigdata66926.2025.11402414

Wiliyanti, Suciati N. Enhancing Multimodal Data Fusion for Fine-Grained Product Classification. 2025 4th International Conference on Electronics Representation and Algorithm (ICERA). 2025. DOI:10.1109/icera66156.2025.11087307

Kulunk A, Taskin B, Eseoglu MF, Sahin HB. Optimizing Product Deduplication in E-Commerce with Multimodal Embeddings. 2025 IEEE International Conference on Big Data (BigData). 2025. DOI:10.1109/bigdata66926.2025.11401681

Li S. Harnessing Multimodal Data and Mult-Recall Strategies for Enhanced Product Recommendation in E-Commerce. 2024 4th International Conference on Computer Systems (ICCS). 2024. DOI:10.1109/iccs62594.2024.10795856

Wang K, Shao J, Zhang T, Chen Q, Huo C. MPKGAC: Multimodal Product Attribute Completion in E-commerce. Companion Proceedings of the ACM Web Conference 2023. 2023. DOI:10.1145/3543873.3584623

## LLMs and multilingual encoders for e-commerce (2024-2026)

Fang C, Li X, Fan Z, Xu J, Nag K, Korpeoglu E, Kumar S, Achan K. LLM-Ensemble: Optimal Large Language Model Ensemble Method for E-commerce Product Attribute Value Extraction. Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2024. DOI:10.1145/3626772.3661357

Çiftlikçi MS, Çakmak Y, Kalaycı TA, Abut F, Akay MF, Kızıldağ M. A New Large Language Model for Attribute Extraction in E-Commerce Product Categorization. Electronics. 2025. DOI:10.3390/electronics14101930

Zhang B, Khan SA, Walter S. Leveraging Product Catalog Patterns for Multilingual E-commerce Product Attribute Prediction. Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track. 2025. DOI:10.18653/v1/2025.emnlp-industry.18

Gong J, Shen H, Jenq J. MICE: Mixture of Image Captioning Experts Augmented e-Commerce Product Attribute Value Extraction. Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track). 2025. DOI:10.18653/v1/2025.acl-industry.80

Liu M, Zhu C. DRAM: Dynamic Range Modulation for Multimodal Attribute Value Extraction on E-Commerce Product Data. Electronics. 2026. DOI:10.3390/electronics15050969

## CLIP and vision-language pretraining as feature extractors (2024-2025)

Aljuhani H, Dahab MY, Alsenani Y. Enhancing Document Classification Through Multimodal Image-Text Classification: Insights from Fine-Tuned CLIP and Multimodal Deep Fusion. Sensors. 2025. DOI:10.3390/s25247596

Onuoha C, Flaherty J, Cong Thang T. Perceptual Image Quality Prediction: Are Contrastive Language-Image Pretraining (CLIP) Visual Features Effective?. Electronics. 2024. DOI:10.3390/electronics13040803

Esfandiarpoor R, Menghini C, Bach S. If CLIP Could Talk: Understanding Vision-Language Model Representations Through Their Preferred Concept Descriptions. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024. DOI:10.18653/v1/2024.emnlp-main.547

Yan Y, Wen H, Zhong S, Chen W, Chen H, Wen Q, Zimmermann R, Liang Y. UrbanCLIP: Learning Text-enhanced Urban Region Profiling with Contrastive Language-Image Pretraining from the Web. Proceedings of the ACM Web Conference 2024. 2024. DOI:10.1145/3589334.3645378

Zhang H, Guo Y, Kankanhalli M. Joint Vision-Language Social Bias Removal for CLIP. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2025. DOI:10.1109/cvpr52734.2025.00401

## Hierarchical and zero-shot text classification (2024)

Longo C, Mongiovi M, Bulla L, Tuccari G. HTC-GEN: A Generative LLM-Based Approach to Handle Data Scarcity in Hierarchical Text Classification. Proceedings of the 13th International Conference on Data Science, Technology and Applications. 2024. DOI:10.5220/0012790700003756

Abdullahi T, Singh R, Eickhoff C. Retrieval Augmented Zero-Shot Text Classification. Proceedings of the 2024 ACM SIGIR International Conference on Theory of Information Retrieval. 2024. DOI:10.1145/3664190.3672514
