![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-multimodal-orange) ![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)

# Rakuten E-commerce Multimodal Product Classification

Classifies French e-commerce product listings into 27 categories using text (product title+description) and image fusion.

---

## Task

**Multimodal Classification (CV + NLP)**

---

## Architecture

```
Product Title+Image → CamemBERT (text) + ResNet50 (vision) → Late Fusion → 27-class Output
```

---

## Key Features

- 27-class product classification from text + image
- CamemBERT for French product title/description encoding
- ResNet-50 image backbone for product photo features
- Late-fusion ensemble (text score + image score weighted average)
- Weighted F1 evaluation matching the Rakuten challenge metric

---

## Dataset

[Rakuten France Multimodal Product Classification Challenge](https://challengedata.ens.fr/challenges/35)

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/rakuten-multimodal-classifier.git
cd rakuten-multimodal-classifier
pip install -r requirements.txt

# See data/README.md for dataset download
jupyter notebook notebooks/01_eda.ipynb
# or run modeling:
jupyter notebook notebooks/03_modeling.ipynb
jupyter notebook notebooks/03_modeling.ipynb  # advanced model (GPU recommended)
```

---

## Tech Stack

`PyTorch · CLIP · CamemBERT · torchvision · scikit-learn`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Germany.

---

## License

MIT
