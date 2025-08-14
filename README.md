# Computer-Aided Breast Cancer Detection via Ensemble of Texture & Statistical Image Features

**Authors:** Soumya Deep Roy, Soham Das, Devroop Kar, Friedhelm Schwenker, Ram Sarkar

Published in *Sensors* (2021). Based on ensembling texture-based and statistical features for breast cancer image classification.

---

##  Repository Overview

| File/Folder | Purpose |
|-------------|---------|
| `*.ipynb` (Notebook) | End-to-end workflow: data exploration, feature extraction, model stacking, and evaluation |
| `glcm_kaggle2.py`, `nfeat2.py` | Scripts for extracting GLCM/statistical features and building the feature set |
| `flowchart_new.png`, `stacking_new.png`, `stacking_new2.png` | Visual diagrams illustrating the methodology and stacking approach |
| `README.md` | Project overview, instructions, and citation |

---

##  Quick Start

### Prerequisites
- Python 3.7+
- Jupyter
- Required libraries (NumPy, pandas, scikit-learn, OpenCV, Matplotlib, etc.)

### Setup
```bash
git clone <this repo>
cd Computer-Aided-Breast-Cancer-Detection-using-Ensembling-of-Texture-and-Statistical-Image-Features
pip install -r requirements.txt  # if you add one
jupyter notebook
