# RNA 3D Structure Prediction Pipeline 🧬

Advanced machine learning approaches for predicting RNA tertiary structures using ensemble strategies and thermal perturbation methods.

## 📊 Overview

This repository contains novel implementations for RNA 3D structure prediction, developed as part of research into the Stanford RNA 3D Folding challenge. Our approaches combine multiple machine learning techniques with biophysically-inspired perturbation strategies to generate diverse and accurate structural predictions.

## 🚀 Key Features

- **Multi-Seed Ensemble Strategy**: Leverages Random Forest models with multiple random seeds (123, 42, 789) to capture different structural perspectives
- **Thermal Ensemble Kabsch Method**: Incorporates temperature-based perturbations with Kabsch alignment for physically realistic structure variations
- **Optimized for Kaggle Environment**: Ready-to-run implementations for the Stanford RNA 3D Folding competition
- **Comprehensive Pipeline**: From sequence processing to 3D coordinate prediction with visualization

## 🔬 Methods

### 1. Multi-Seed Ensemble
Generates diverse structural predictions by training Random Forest models with different random seeds, capturing the stochastic nature of RNA folding landscapes.

### 2. Thermal Perturbation with Kabsch Alignment
Simulates molecular vibrations at different temperature scales (0.25x, 0.45x, 0.75x) using various noise distributions (Laplace, Student's t, Normal) to explore conformational space.

## 📁 Repository Structure

```
rna-3d-structure-prediction/
│
├── 📂 data/
│   └── (train/test CSV'leri)
│
├── 📂 exploratory-analysis/
│   ├── EDA.ipynb
│   └── EDA_2.ipynb
│
├── 📂 models/
│   ├── 📂 baseline-models/
│   │   ├── model_prediction_first_try_upgraded_0179.py
│   │   ├── Model_prediction_random_forest_02_0167.py
│   │   ├── Model_prediction_random_forest_0178.py
│   │   └── Model_prediction_xgboost.py
│   │
│   ├── 📂 ensemble-strategies/
│   │   ├── Model_Prediction_random_forest_upgraded_seed123_42_789_Çoklu Tohum Ensemble Stratejisi.py
│   │   ├── Model_Prediction_random_forest_upgraded_seed42_0180.py
│   │   ├── Model_Prediction_random_forest_upgraded_seed123_0183.py
│   │   ├── Model_Prediction_random_forest_upgraded_seed555_0179.py
│   │   └── Model_Prediction_random_forest_upgraded_seed789_0180.py
│   │
│   ├── 📂 biological-filters/
│   │   ├── İyileştirilmiş Biyolojik Filtreleme ile Termal Ensemble Kabsch_0.200.py
│   │   ├── İyileştirilmiş Biyolojik Filtreleme ve Dolaylık Önleme_0.203.py
│   │   ├── İyileştirilmiş Biyolojik Filtreleme ve Koaksiyel İstifleme_0.204.py
│   │   └── İyileştiriliş Watson-Crick Olmayan Baz Çiftleri Tespiti_0.205.py
│   │
│   ├── 📂 thermal-ensemble/
│   │   ├── Temel Termal Ensemble Kabsch_0.198.py
│   │   └── Termal Ensemble Kabsch_0.197.py
│   │
│   ├── 📂 advanced-strategies/
│   │   ├── Geliştirilimiş_Seed_123_Extreme_Varyasyon_Stratejisi_seed123_0190.py
│   │   ├── Kabsch-Geliştirilimiş Korelasyonlu Gürültü_0.196.py
│   │   ├── İnce Ayarlı Korelasyonlu Gürültü Stratejisi_0.194.py
│   │   └── RNA Uzunluğuna Adaptif Parametre Yaklaşımı_İnce Ayarlı_0.205.py
│   │
│   ├── 📂 seed123-variations/
│   │   ├── seed123_Agresif Gürültü Spektrumu Stratejisi_0.187.py
│   │   ├── seed123_Gelişmiş Biyolojik Entegrasyon Stratejisi_0.190.py
│   │   ├── seed123_Korelasyonlu Gürültü Tüm Yapılarda_0.194.py
│   │   ├── seed123_RibonanzaNet + Extreme Varyasyon Ensemble Stratejisi.py
│   │   ├── seed123_RNA Motif-Bazlı Varyasyon Stratejisi_0.192.py
│   │   ├── seed123_Student's t-Dağılımı Stratejisi0.190.py
│   │   └── seed123_Yapılar Arası Korelasyon Stratejisi_0.193.py
│   │
│   └── 📂 optimization/
│       ├── Model_prediction_random_optimized.py
│       └── Model_Prediction_random_forest_hibrit_03_0170.py
│
├── 📂 preprocessing/
│   └── Pre-Processing.ipynb
│
├── 📂 results/
│   ├── 📂 foto/
│   │   └── (Görselleştirmeler)
│   ├── 📂 submissions/
│   │   └── fist solution/ (İlk çözümünüz)
│   └── 📂 error_logs/
│       └── (Hata logları)
│
├── README.md
```

📈 Performance
Our ensemble approach achieves competitive results on the Stanford RNA 3D Folding dataset, with particular strength in capturing alternative conformational states.
📚 Background
This work builds upon insights from RNA-Puzzles Round V (Nature Methods) and incorporates lessons learned from CASP competitions. The models address key challenges in RNA structure prediction:

Identification of non-Watson-Crick base pairs
Correct coaxial stacking between helices
Avoidance of structural entanglements

🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for discussion.

