# Computer-Aided Breast Cancer Detection via Ensemble of Texture & Statistical Image Features

**Authors:** Soumya Deep Roy, Soham Das, Devroop Kar, Friedhelm Schwenker, Ram Sarkar  
**Published in:** *Sensors*, Volume 21, Issue 11, Article 3628 (2021)  
**DOI:** [10.3390/s21113628](https://doi.org/10.3390/s21113628)  
**Paper Link:** [📄 Read the Full Paper on MDPI](https://www.mdpi.com/1424-8220/21/11/3628)

---

## 📖 Abstract

Breast cancer remains one of the leading causes of mortality worldwide. Early and accurate detection is essential to improving survival rates. This work presents an ensemble-based CAD (Computer-Aided Detection) system that combines **texture-based** and **statistical image features** to classify IDC (Invasive Ductal Carcinoma) images. Using a dataset of 277,524 histopathological images, we extract **782 fused features** (SIFT, SURF, ORB + Haralick descriptors) and train a **stacked ensemble** of classifiers (Random Forest, Extra Trees, XGBoost, AdaBoost, CatBoost, MLP), improving robustness and classification accuracy.  

Full details can be found in the published paper: [📄 MDPI Sensors 2021](https://www.mdpi.com/1424-8220/21/11/3628).

---

## 📂 Repository Overview

| File/Folder | Purpose |
|-------------|---------|
| `Computer Aided Breast Cancer Detection...ipynb` | End-to-end Jupyter Notebook implementation |
| `glcm_kaggle2.py` | Extracts GLCM texture features |
| `nfeat2.py` | Extracts statistical image features |
| `flowchart_new.png` | Workflow diagram |
| `stacking_new.png` / `stacking_new2.png` | Visual explanation of the stacking ensemble |
| `README.md` | Project documentation |

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Install required Python libraries: NumPy, pandas, scikit-learn, OpenCV, Matplotlib, XGBoost, CatBoost, etc.

### Installation
```bash
git clone https://github.com/sdrxc/Computer-Aided-Breast-Cancer-Detection-using-Ensembling-of-Texture-and-Statistical-Image-Features.git
cd Computer-Aided-Breast-Cancer-Detection-using-Ensembling-of-Texture-and-Statistical-Image-Features
pip install -r requirements.txt  # if available
jupyter notebook
