"""Generate the modeling notebook (nbformat=4) for project 06_rakuten_multimodal.

Run: python3 build_modeling_nb.py
Produces: 03_modeling.ipynb (unexecuted). Then execute via jupyter nbconvert.
"""
import nbformat as nbf
from pathlib import Path

NB_PATH = Path(__file__).parent / "03_modeling.ipynb"

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Rakuten Multimodal Product Classification: Modeling\n"
    "\n"
    "
    "\n"
    "**Goal:** classify `prdtypecode` (27 classes) on a 30k stratified subsample of\n"
    "the 84,916-row training set. Two models:\n"
    "\n"
    "1. **Baseline (text-only):** TF-IDF (1-2 grams, max_features=20,000) on\n"
    "   `designation` + cleaned `description` -> Logistic Regression.\n"
    "2. **Improved (multimodal):** baseline TF-IDF features concatenated with\n"
    "   ResNet18 (ImageNet1k) image embeddings (224x224, streamed from\n"
    "   `images.zip`) -> Logistic Regression on the late-fusion vector.\n"
    "\n"
    "Metrics: weighted F1, macro-F1, top-1 accuracy, top-3 accuracy, plus the\n"
    "top confused class pairs read off the confusion matrix.\n"
    "\n"
    "Images are streamed via `zipfile.ZipFile.open` - the 2.5 GB archive is\n"
    "never extracted to disk."
))

cells.append(nbf.v4.new_markdown_cell("## 1. Imports, paths, configuration"))

cells.append(nbf.v4.new_code_cell(
    "import os, io, re, json, time, zipfile, warnings, pickle\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import scipy.sparse as sp\n"
    "from PIL import Image\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.feature_extraction.text import TfidfVectorizer\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n"
    "from sklearn.metrics import (f1_score, accuracy_score, top_k_accuracy_score,\n"
    "                             confusion_matrix, classification_report)\n"
    "\n"
    "import torch\n"
    "from torch.utils.data import DataLoader, Dataset\n"
    "from torchvision import transforms, models\n"
    "\n"
    "ROOT = Path('..').resolve()\n"
    "DATA = ROOT / 'data'\n"
    "DELIV = ROOT / 'deliverables'\n"
    "REPORTS = ROOT / 'reports'\n"
    "DELIV.mkdir(exist_ok=True)\n"
    "REPORTS.mkdir(exist_ok=True)\n"
    "\n"
    "X_TRAIN = DATA / 'X_train_update.csv'\n"
    "Y_TRAIN = DATA / 'Y_train_CVw08PX.csv'\n"
    "IMG_ZIP = DATA / 'images.zip'\n"
    "for p in [X_TRAIN, Y_TRAIN, IMG_ZIP]:\n"
    "    assert p.exists(), p\n"
    "\n"
    "SEED        = 42\n"
    "SUBSAMPLE_N = 30_000\n"
    "TFIDF_MAX   = 20_000\n"
    "IMG_SIZE    = 224\n"
    "BATCH_SIZE  = 64\n"
    "DEVICE      = torch.device('cpu')\n"
    "np.random.seed(SEED); torch.manual_seed(SEED)\n"
    "print('config ok | device:', DEVICE)"
))

cells.append(nbf.v4.new_markdown_cell("## 2. Load CSVs and join on row index"))

cells.append(nbf.v4.new_code_cell(
    "x = pd.read_csv(X_TRAIN, index_col=0)\n"
    "y = pd.read_csv(Y_TRAIN, index_col=0)\n"
    "df = x.join(y)\n"
    "print('full train:', df.shape)\n"
    "print('classes   :', df['prdtypecode'].nunique())\n"
    "df.head(3)"
))

cells.append(nbf.v4.new_markdown_cell("## 3. Stratified subsample to 30k rows"))

cells.append(nbf.v4.new_code_cell(
    "frac = SUBSAMPLE_N / len(df)\n"
    "sub, _ = train_test_split(df, train_size=frac, stratify=df['prdtypecode'],\n"
    "                          random_state=SEED)\n"
    "sub = sub.reset_index(drop=True)\n"
    "print('subsample:', sub.shape)\n"
    "print('classes in subsample:', sub['prdtypecode'].nunique())\n"
    "print('class size range:', sub['prdtypecode'].value_counts().min(),\n"
    "      '->', sub['prdtypecode'].value_counts().max())"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 4. Text cleaning: strip HTML, merge designation + description"
))

cells.append(nbf.v4.new_code_cell(
    "_html = re.compile(r'<[^>]+>')\n"
    "_ws   = re.compile(r'\\s+')\n"
    "def clean_text(s):\n"
    "    if not isinstance(s, str):\n"
    "        return ''\n"
    "    s = _html.sub(' ', s)\n"
    "    s = s.replace('&nbsp;', ' ').replace('&amp;', '&')\n"
    "    s = _ws.sub(' ', s).strip().lower()\n"
    "    return s\n"
    "\n"
    "sub['text'] = (sub['designation'].fillna('').map(clean_text) + ' '\n"
    "               + sub['description'].fillna('').map(clean_text)).str.strip()\n"
    "print('empty text rows:', (sub['text'].str.len() == 0).sum())\n"
    "print('mean text chars:', round(sub['text'].str.len().mean(), 1))\n"
    "sub[['text', 'prdtypecode']].head(2)"
))

cells.append(nbf.v4.new_markdown_cell("## 5. Train / test split (80 / 20, stratified)"))

cells.append(nbf.v4.new_code_cell(
    "le = LabelEncoder()\n"
    "y_all = le.fit_transform(sub['prdtypecode'].values)\n"
    "n_classes = len(le.classes_)\n"
    "print('n_classes:', n_classes)\n"
    "\n"
    "idx_train, idx_test = train_test_split(\n"
    "    np.arange(len(sub)), test_size=0.2, stratify=y_all, random_state=SEED)\n"
    "print('train:', len(idx_train), 'test:', len(idx_test))\n"
    "\n"
    "y_train = y_all[idx_train]\n"
    "y_test  = y_all[idx_test]"
))

cells.append(nbf.v4.new_markdown_cell("## 6. Text-only baseline (TF-IDF + Logistic Regression)"))

cells.append(nbf.v4.new_code_cell(
    "t0 = time.time()\n"
    "tfidf = TfidfVectorizer(max_features=TFIDF_MAX, ngram_range=(1, 2),\n"
    "                        min_df=2, sublinear_tf=True, strip_accents='unicode')\n"
    "X_text_train = tfidf.fit_transform(sub['text'].iloc[idx_train])\n"
    "X_text_test  = tfidf.transform(sub['text'].iloc[idx_test])\n"
    "print('TF-IDF shape:', X_text_train.shape, '|', round(time.time()-t0, 1), 's')\n"
    "\n"
    "t0 = time.time()\n"
    "clf_text = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1,\n"
    "                              solver='liblinear', class_weight='balanced')\n"
    "clf_text.fit(X_text_train, y_train)\n"
    "print('LogReg fit:', round(time.time()-t0, 1), 's')"
))

cells.append(nbf.v4.new_code_cell(
    "def score(y_true, y_pred, y_proba, n_classes, tag):\n"
    "    out = {\n"
    "        'tag'        : tag,\n"
    "        'top1_acc'   : float(accuracy_score(y_true, y_pred)),\n"
    "        'top3_acc'   : float(top_k_accuracy_score(y_true, y_proba, k=3,\n"
    "                                                  labels=np.arange(n_classes))),\n"
    "        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted')),\n"
    "        'macro_f1'   : float(f1_score(y_true, y_pred, average='macro')),\n"
    "    }\n"
    "    print(f\"[{tag}] top1={out['top1_acc']:.4f} top3={out['top3_acc']:.4f} \"\n"
    "          f\"weighted_f1={out['weighted_f1']:.4f} macro_f1={out['macro_f1']:.4f}\")\n"
    "    return out\n"
    "\n"
    "y_pred_text  = clf_text.predict(X_text_test)\n"
    "y_proba_text = clf_text.predict_proba(X_text_test)\n"
    "metrics_text = score(y_test, y_pred_text, y_proba_text, n_classes, 'text-only')"
))

cells.append(nbf.v4.new_code_cell(
    "# Top confused class pairs (off-diagonal mass) for the text-only model.\n"
    "def top_confusions(y_true, y_pred, le, k=10):\n"
    "    cm = confusion_matrix(y_true, y_pred)\n"
    "    pairs = []\n"
    "    for i in range(cm.shape[0]):\n"
    "        for j in range(cm.shape[1]):\n"
    "            if i != j and cm[i, j] > 0:\n"
    "                pairs.append((int(le.classes_[i]), int(le.classes_[j]), int(cm[i, j])))\n"
    "    pairs.sort(key=lambda t: -t[2])\n"
    "    return pairs[:k]\n"
    "\n"
    "conf_text = top_confusions(y_test, y_pred_text, le, k=10)\n"
    "print('top 10 confused (true -> pred, count):')\n"
    "for tup in conf_text:\n"
    "    print(' ', tup)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 7. Image features: ResNet18 (ImageNet1k) embeddings streamed from zip"
))

cells.append(nbf.v4.new_code_cell(
    "# Build a list of in-zip paths for each subsampled row.\n"
    "def img_member(row):\n"
    "    return f\"images/image_train/image_{row['imageid']}_product_{row['productid']}.jpg\"\n"
    "\n"
    "members = [img_member(r) for _, r in sub[['imageid', 'productid']].iterrows()]\n"
    "print('first member:', members[0])\n"
    "print('total members to embed:', len(members))"
))

cells.append(nbf.v4.new_code_cell(
    "# Streaming dataset: opens images.zip in __getitem__ to stay fork-safe\n"
    "# (we use num_workers=0 so a single handle is fine, but this pattern is\n"
    "# robust either way).\n"
    "tfm = transforms.Compose([\n"
    "    transforms.Resize((IMG_SIZE, IMG_SIZE)),\n"
    "    transforms.ToTensor(),\n"
    "    transforms.Normalize(mean=[0.485, 0.456, 0.406],\n"
    "                         std=[0.229, 0.224, 0.225]),\n"
    "])\n"
    "\n"
    "class ZipImageDataset(Dataset):\n"
    "    def __init__(self, zip_path, members, transform):\n"
    "        self.zip_path = str(zip_path)\n"
    "        self.members  = members\n"
    "        self.transform = transform\n"
    "        self._zf = None\n"
    "    def _open(self):\n"
    "        if self._zf is None:\n"
    "            self._zf = zipfile.ZipFile(self.zip_path, 'r')\n"
    "    def __len__(self):\n"
    "        return len(self.members)\n"
    "    def __getitem__(self, i):\n"
    "        self._open()\n"
    "        try:\n"
    "            with self._zf.open(self.members[i]) as f:\n"
    "                im = Image.open(io.BytesIO(f.read())).convert('RGB')\n"
    "        except KeyError:\n"
    "            im = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (255, 255, 255))\n"
    "        return self.transform(im)\n"
    "\n"
    "ds = ZipImageDataset(IMG_ZIP, members, tfm)\n"
    "dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)\n"
    "print('dataset size:', len(ds))"
))

cells.append(nbf.v4.new_code_cell(
    "# ResNet18 feature extractor: drop the classification head, keep the 512-d pool.\n"
    "rn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)\n"
    "rn.fc = torch.nn.Identity()\n"
    "rn.eval().to(DEVICE)\n"
    "for p in rn.parameters():\n"
    "    p.requires_grad_(False)\n"
    "print('resnet18 ready, output dim = 512')"
))

cells.append(nbf.v4.new_code_cell(
    "# Forward all 30k images on CPU. ResNet18 + 224x224 is light enough\n"
    "# to keep this under ~10 minutes on a modern laptop CPU.\n"
    "feats = np.zeros((len(ds), 512), dtype=np.float32)\n"
    "n_done = 0\n"
    "t0 = time.time()\n"
    "with torch.no_grad():\n"
    "    for batch in dl:\n"
    "        batch = batch.to(DEVICE, non_blocking=True)\n"
    "        out = rn(batch).cpu().numpy().astype(np.float32)\n"
    "        feats[n_done:n_done + out.shape[0]] = out\n"
    "        n_done += out.shape[0]\n"
    "        if n_done % (BATCH_SIZE * 20) == 0:\n"
    "            rate = n_done / (time.time() - t0 + 1e-6)\n"
    "            print(f'  {n_done}/{len(ds)}  ({rate:.1f} img/s)')\n"
    "print('image features:', feats.shape, '|', round(time.time()-t0, 1), 's total')"
))

cells.append(nbf.v4.new_markdown_cell("## 8. Multimodal late fusion: concat TF-IDF + image features"))

cells.append(nbf.v4.new_code_cell(
    "# Standardize the dense image features so they share scale with TF-IDF rows.\n"
    "scaler = StandardScaler(with_mean=False)\n"
    "feats_train = scaler.fit_transform(feats[idx_train])\n"
    "feats_test  = scaler.transform(feats[idx_test])\n"
    "\n"
    "X_mm_train = sp.hstack([X_text_train, sp.csr_matrix(feats_train)]).tocsr()\n"
    "X_mm_test  = sp.hstack([X_text_test,  sp.csr_matrix(feats_test)]).tocsr()\n"
    "print('multimodal shape:', X_mm_train.shape)\n"
    "\n"
    "t0 = time.time()\n"
    "clf_mm = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1,\n"
    "                            solver='liblinear', class_weight='balanced')\n"
    "clf_mm.fit(X_mm_train, y_train)\n"
    "print('multimodal LogReg fit:', round(time.time()-t0, 1), 's')\n"
    "\n"
    "y_pred_mm  = clf_mm.predict(X_mm_test)\n"
    "y_proba_mm = clf_mm.predict_proba(X_mm_test)\n"
    "metrics_mm = score(y_test, y_pred_mm, y_proba_mm, n_classes, 'multimodal')\n"
    "\n"
    "conf_mm = top_confusions(y_test, y_pred_mm, le, k=10)\n"
    "print('top 10 confused (multimodal):')\n"
    "for tup in conf_mm:\n"
    "    print(' ', tup)"
))

cells.append(nbf.v4.new_markdown_cell("## 9. Per-class report and improvement summary"))

cells.append(nbf.v4.new_code_cell(
    "report_text = classification_report(\n"
    "    y_test, y_pred_text, target_names=[str(c) for c in le.classes_],\n"
    "    digits=3, zero_division=0)\n"
    "report_mm = classification_report(\n"
    "    y_test, y_pred_mm,   target_names=[str(c) for c in le.classes_],\n"
    "    digits=3, zero_division=0)\n"
    "print('=== text-only ==='); print(report_text)\n"
    "print('=== multimodal ==='); print(report_mm)"
))

cells.append(nbf.v4.new_code_cell(
    "delta = {\n"
    "    'top1_delta'        : metrics_mm['top1_acc']    - metrics_text['top1_acc'],\n"
    "    'top3_delta'        : metrics_mm['top3_acc']    - metrics_text['top3_acc'],\n"
    "    'weighted_f1_delta' : metrics_mm['weighted_f1'] - metrics_text['weighted_f1'],\n"
    "    'macro_f1_delta'    : metrics_mm['macro_f1']    - metrics_text['macro_f1'],\n"
    "}\n"
    "print('multimodal improvement over text-only:')\n"
    "for k, v in delta.items():\n"
    "    print(f'  {k:20s} {v:+.4f}')"
))

cells.append(nbf.v4.new_markdown_cell("## 10. Persist deliverables: pickle + metrics.json"))

cells.append(nbf.v4.new_code_cell(
    "payload = {\n"
    "    'tfidf'         : tfidf,\n"
    "    'scaler_image'  : scaler,\n"
    "    'label_encoder' : le,\n"
    "    'clf_text'      : clf_text,\n"
    "    'clf_multimodal': clf_mm,\n"
    "    'config': {\n"
    "        'subsample_n' : SUBSAMPLE_N,\n"
    "        'tfidf_max'   : TFIDF_MAX,\n"
    "        'img_size'    : IMG_SIZE,\n"
    "        'image_backbone': 'resnet18.imagenet1k_v1',\n"
    "        'fusion'      : 'late_concat_logreg',\n"
    "        'seed'        : SEED,\n"
    "    },\n"
    "}\n"
    "out_pkl = DELIV / 'rakuten_textimg.pkl'\n"
    "with open(out_pkl, 'wb') as f:\n"
    "    pickle.dump(payload, f)\n"
    "print('wrote', out_pkl, '|', round(out_pkl.stat().st_size / 1e6, 2), 'MB')\n"
    "\n"
    "metrics_payload = {\n"
    "    'text_only'  : metrics_text,\n"
    "    'multimodal' : metrics_mm,\n"
    "    'delta'      : delta,\n"
    "    'top_confusions_text'      : conf_text,\n"
    "    'top_confusions_multimodal': conf_mm,\n"
    "    'n_classes'  : int(n_classes),\n"
    "    'n_train'    : int(len(idx_train)),\n"
    "    'n_test'     : int(len(idx_test)),\n"
    "    'config': payload['config'],\n"
    "}\n"
    "out_json = DELIV / 'metrics.json'\n"
    "with open(out_json, 'w') as f:\n"
    "    json.dump(metrics_payload, f, indent=2, default=str)\n"
    "print('wrote', out_json)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 11. Summary\n"
    "\n"
    "On the 30,000-row stratified subsample (24,000 train / 6,000 test, 27\n"
    "classes), the text-only TF-IDF + Logistic Regression baseline is the\n"
    "reference. The multimodal model adds ResNet18 ImageNet1k embeddings\n"
    "(512-d) concatenated to the TF-IDF vector and re-fits the same Logistic\n"
    "Regression. Both models use `class_weight='balanced'` to counter the\n"
    "13.4x class imbalance reported in the EDA. Macro-F1 is the headline\n"
    "metric because of that imbalance. See `reports/modeling_1.md` and\n"
    "`reports/modeling_2.md` for the written analysis."
))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
}

NB_PATH.write_text(nbf.writes(nb), encoding='utf-8')
print('wrote', NB_PATH)
