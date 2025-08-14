# Computer-Aided Breast Cancer Detection via Ensemble of Texture & Statistical Image Features

**Authors:** Soumya Deep Roy, Soham Das, Devroop Kar, Friedhelm Schwenker, Ram Sarkar  
**Published in:** *Sensors*, Volume 21, Issue 11, Article 3628 (2021)  
**DOI:** [10.3390/s21113628](https://doi.org/10.3390/s21113628)  
**Paper Link:** [📄 Read the Full Paper on MDPI](https://www.mdpi.com/1424-8220/21/11/3628)

---

## 📖 Abstract

Breast cancer, like most forms of cancer, is a fatal disease that claims more than half a million lives every year. In 2020, breast cancer overtook lung cancer as the most commonly diagnosed form of cancer. Though extremely deadly, the survival rate and longevity increase substantially with early detection and diagnosis. The treatment protocol also varies with the stage of breast cancer. Diagnosis is typically done using histopathological slides from which it is possible to determine whether the tissue is in the Ductal Carcinoma In Situ (DCIS) stage, in which the cancerous cells have not spread into the encompassing breast tissue, or in the Invasive Ductal Carcinoma (IDC) stage, wherein the cells have penetrated into the neighboring tissues. IDC detection is extremely time-consuming and challenging for physicians. Hence, this can be modeled as an image classification task where pattern recognition and machine learning can be used to aid doctors and medical practitioners in making such crucial decisions. In the present paper, we use an IDC Breast Cancer dataset that contains 277,524 images (with 78,786 IDC positive images and 198,738 IDC negative images) to classify the images into IDC(+) and IDC(-). To that end, we use feature extractors, including textural features, such as SIFT, SURF and ORB, and statistical features, such as Haralick texture features. These features are then combined to yield a dataset of 782 features. These features are ensembled by stacking using various Machine Learning classifiers, such as Random Forest, Extra Trees, XGBoost, AdaBoost, CatBoost and Multi Layer Perceptron followed by feature selection using Pearson Correlation Coefficient to yield a dataset with four features that are then used for classification. From our experimental results, we found that CatBoost yielded the highest accuracy (92.55%), which is at par with other state-of-the-art results—most of which employ Deep Learning architectures. The source code is available in the GitHub repository.

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
