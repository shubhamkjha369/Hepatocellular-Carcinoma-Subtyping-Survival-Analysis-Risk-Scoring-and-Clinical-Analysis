<div align="center">

# OncoResolve-Liver: High-Hygiene Explainable AI and Precision Oncology Framework for Hepatocellular Carcinoma Subtyping

### An end-to-end transcriptomics, machine learning, and precision oncology pipeline for classifying Hepatocellular Carcinoma (HCC) molecular subtypes with SHAP explainability and cross-platform multi-cohort external validation.

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn 1.4+](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=flat)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubhamkjha369/Hepatocellular-Carcinoma-Subtyping-Survival-Analysis-Risk-Scoring-and-Clinical-Analysis/blob/main/Liver_Cancer_Subtyping_and_Precision_Profiling.ipynb)

**Computational Biology and Translational Medicine Framework**

</div>

---

## Table of Contents

- [Abstract](#abstract)
- [Project Aim](#project-aim)
- [Primary Discovery & Validation Datasets](#datasets)
- [Methodological Contributions](#methodological-contributions)
- [Pipeline Architecture](#pipeline-architecture)
- [Notebook Structure](#notebook-structure)
- [Section-by-Section Results](#section-results)
- [Key Findings & Validation Outcomes](#key-findings)
- [Results Files](#results-files)
- [Reproduce the Analysis](#reproduce-the-analysis)
- [Technologies](#technologies)
- [Limitations and Future Work](#limitations)

---

<a id="abstract"></a>
## Abstract

Hepatocellular Carcinoma (HCC) is a highly heterogeneous liver cancer characterized by complex transcriptional reprogramming `[LITERATURE-SUPPORTED]`. While computational subtyping from transcriptomics has advanced precision oncology, standard machine learning pipelines are frequently plagued by technical flaws including row-level data leakage, unvalidated feature selections, and poor generalizability across disparate microarray and RNA-seq profiling platforms `[LITERATURE-SUPPORTED]`.

We present **OncoResolve-Liver**, a high-hygiene machine learning and precision oncology pipeline designed to address these challenges. Using a primary discovery cohort of 357 clinical samples from [GSE14520](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/GSE14520-GPL571_series_matrix.txt.gz) (Affymetrix GPL571), we implement an anti-leakage cross-validation protocol where Z-score standardization and a consensus feature selection ensemble utilizing majority voting (ANOVA, LASSO, Random Forest Gini) are fit strictly within each training partition `[DEMONSTRATED]`. This feature selection identifies a stable signature of **50 consensus biomarker genes** `[DEMONSTRATED]`.

Binary classification models (HCC tumor vs. non-tumor/normal liver tissue) are trained using a 5-fold outer, 3-fold inner Stratified Nested Cross-Validation setup `[DEMONSTRATED]`. The champion **Linear SVM** and **Kernel (RBF) SVM** pipelines are evaluated on three independent, cross-platform external validation cohorts `[DEMONSTRATED]`:
1. **[GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz)** (Singapore, Illumina HT-12 microarray, N=167, 115 HCC vs 52 normal)
2. **[TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz)** (RNA-seq, N=372, HCC tumor only)
3. **[GSE36376](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36376/matrix/GSE36376_series_matrix.txt.gz)** (Illumina Ref-8 microarray, N=433, 240 HCC vs 193 normal)

The locked pipelines achieve outstanding diagnostic transferability `[INTERPRETATION]`:
* **[GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz)**:
  - **Linear SVM**: Accuracy = **92.8%**, Macro F1 = **91.9%**, AUC-ROC = **97.5%** `[DEMONSTRATED]`
  - **Non-Linear (Kernel SVM)**: Accuracy = **94.0%**, Macro F1 = **93.2%**, AUC-ROC = **97.6%** `[DEMONSTRATED]`
* **[GSE36376](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36376/matrix/GSE36376_series_matrix.txt.gz)**:
  - **Linear SVM**: Accuracy = **97.9%**, Macro F1 = **97.9%**, AUC-ROC = **98.6%** `[DEMONSTRATED]`
  - **Non-Linear (Kernel SVM)**: Accuracy = **97.7%**, Macro F1 = **97.7%**, AUC-ROC = **98.9%** `[DEMONSTRATED]`
* **[TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz)**:
  - **Linear SVM**: Sensitivity/Recall = **61.2%** `[DEMONSTRATED]`
  - **Non-Linear (Kernel SVM)**: Sensitivity/Recall = **71.6%** `[DEMONSTRATED]`

Decisions are mapped globally and locally using SHAP attributions, identifying key molecular drivers [DEMONSTRATED]. Finally, we deconvolve the tumor microenvironment (TME) via ssGSEA [DEMONSTRATED], build Cox proportional hazards models on both discovery (GSE14520) and validation cohorts (TCGA-LIHC OS/DFS, GSE76427 RFS) [DEMONSTRATED], and query DepMap CRISPR and LINCS L1000 databases to connect predictive drivers with potential therapeutic dependencies [DEMONSTRATED].

<a id="project-aim"></a>
## Project Aim

OncoResolve-Liver is designed to address six specific technical and clinical objectives:

1. **Anti-leakage classification** — Deploy a finalized **Linear SVM + Kernel SVM** dual-model pipeline trained on GSE14520 where `StandardScaler` and ensemble feature selection are fit strictly *inside* each cross-validation fold—eliminating feature-selection leakage `[DEMONSTRATED]`.
2. **50-gene consensus biomarker discovery** — Identify a stable, biologically validated set of **50 consensus genes** via a tri-method ensemble selector (ANOVA F-test + LASSO L1 + Random Forest Gini). Explain predictions using both **LinearSHAP** and **KernelSHAP** `[DEMONSTRATED]`.
3. **Cross-platform validation on 3 independent external cohorts** — Evaluate the completely locked discovery pipeline (no retraining) on **[GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz)**, **[TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz)**, and **[GSE36376](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36376/matrix/GSE36376_series_matrix.txt.gz)**. Cross-platform transfer requires per-cohort independent Z-score harmonization and strict feature alignment `[DEMONSTRATED]`.
4. **Tumor Microenvironment (TME) Deconvolution** — Estimate immune and stromal cell fractions using ssGSEA via `decoupler` on raw probe levels to map CD8+ T cells, NK cells, B cells, macrophages, CAFs, and endothelial populations `[DEMONSTRATED]`.
5. **Survival & Therapeutic Discovery** — Model overall and recurrence-free survival using Cox Proportional Hazards regression on both discovery and validation datasets (transferring L2-regularized Ridge CRS models) and identify potential therapeutic options by querying DepMap CRISPR essentiality and LINCS L1000 drug discovery datasets `[DEMONSTRATED]`.

---

<a id="datasets"></a>
## Primary Discovery & Validation Datasets

### 1. Primary Discovery Cohort: [GSE14520](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/GSE14520-GPL571_series_matrix.txt.gz) (Affymetrix U133A)
* **Platform**: Affymetrix Human Genome U133A (GPL571)
* **N (clinical)**: **357** clinical samples (181 HCC vs 176 normal controls, after removing cell lines)
* **Features**: 22,279 probes (mapped to HUGO symbols)
* **Survival Data**: OS (Survival status / months) and RFS (Recurr status / months)
* **Direct Download Links**:
  - [Expression Series Matrix (GPL571)](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/GSE14520-GPL571_series_matrix.txt.gz)
  - [Clinical Extra Supplement](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/suppl/GSE14520_Extra_Supplement.txt.gz)
* **Source / Metadata Page**: [GEO GSE14520](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14520)

### 2. External Validation Cohort 1: [GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz) (Illumina HT-12 v4)
* **Platform**: Illumina HumanHT-12 v4 expression beadchip
* **N**: **167** clinical samples (115 HCC vs 52 normal)
* **Survival Data**: RFS years, Recurrence events (modeled from staging)
* **Clinical Stages**: BCLC Stage, TNM Stage, Age, Gender
* **Direct Download Link**: [Expression Series Matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz)
* **Source / Metadata Page**: [GEO GSE76427](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76427)

### 3. External Validation Cohort 2: [TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz) (RNA-seq)
* **Platform**: RNA-seq (Illumina HiSeq RSEM values)
* **N**: **372** clinical samples (HCC primary tumor only)
* **Survival Data**: OS (OS status / months) and DFS (DFS status / months)
* **Clinical Stages**: AJCC Pathologic Stage, Age, Gender
* **Direct Download Link**: [cBioPortal Study Data Archive (Pan-Cancer 2018)](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz)
* **Source / Metadata Page**: [cBioPortal TCGA-LIHC Study](https://www.cbioportal.org/study/summary?id=lihc_tcga_pan_can_atlas_2018)

### 4. External Validation Cohort 3: [GSE36376](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36376/matrix/GSE36376_series_matrix.txt.gz) (Illumina Ref-8)
* **Platform**: Illumina HumanRef-8 v3.0 expression beadchip (GPL10558)
* **N**: **433** clinical samples (240 HCC vs 193 normal)
* **Survival Data**: Staging only (AJCC Stage)
* **Direct Download Links**:
  - [Expression Series Matrix](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36376/matrix/GSE36376_series_matrix.txt.gz)
  - [Platform Annotation (GPL10558)](https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL10nnn/GPL10558/annot/GPL10558.annot.gz)
* **Source / Metadata Page**: [GEO GSE36376](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36376)

---

<a id="methodological-contributions"></a>
## Methodological Contributions

OncoResolve-Liver implements four core methodological contributions:

1. **Strict Nested Cross-Validation Hygiene Framework**: The StandardScaler and all feature selections are fit *strictly within each cross-validation fold*—no global normalization is applied before splitting, avoiding information leak `[DEMONSTRATED]`.
2. **Tri-Method Consensus Feature Selection**: Feature selection uses a consensus vote across three distinct mathematical paradigms:
   - **ANOVA F-Test** (univariate significance)
   - **LASSO (L1 Regularized Logistic Regression)** (sparsity-inducing coefficient shrinkage)
   - **Random Forest Gini Importance** (non-linear ensemble importance)
   A gene is selected for downstream classification only if it is nominated by **at least two of the three methods**, reducing noise `[DEMONSTRATED]`.
3. **Platform-Independent Validation**: Evaluation on external validation cohorts is performed using a **completely locked model** with Z-score harmonization applied independently to each cohort `[DEMONSTRATED]`.

---

<a id="pipeline-architecture"></a>
## Pipeline Architecture

```
RAW DATA: GSE14520 U133A
357 samples x 22,279 probes (Affymetrix GPL571)
              |
              v
SECTION 1: Data Ingestion & QC
  - Remove technical AFFX control probes
  - log2 scale verification, memory downcasting (float32)
  - Stratified 80/20 train/test partition (Discovery vs Holdout)
              |
              v
SECTION 2: Quantile Normalization & PCA Outlier Audit
  - Fit QuantileNormalizer reference ONLY on Discovery split
  - Transform Discovery and Holdout splits using locked reference
  - Detect outliers (mean correlation < -2 SD) and project on PCA space
              |
              v
SECTION 3: Unsupervised Latent Space (PCA, t-SNE, UMAP)
  - Scale features and reduce dimensions using top 5,000 variable probes
  - Visualize sample clusters side-by-side
              |
              v
SECTION 4: Differential Gene Expression (DGE) Profiling
  - Welch's t-test with Benjamini-Hochberg FDR correction
  - Define top significant DEGs (|log2FC| > 1.0, FDR < 0.05)
              |
              v
SECTION 5: Unsupervised Clustering Validation
  - Run K-Means and Hierarchical Ward linkage
  - Evaluate cluster quality using ARI and NMI metrics
              |
              v
SECTION 6: Co-expression Network Topology
  - Construct co-expression adjacency matrix (|r| > 0.85) on top 500 variable probes
  - Calculate node degree and extract top Hub genes
              |
              v
SECTION 7: Ensemble Feature Selection (Strictly inside folds)
  - Nominates features via ANOVA, LASSO, and Random Forest Gini Importance
  - Selects 50 consensus biomarkers
              |
              v
SECTION 8: Repeated Stratified Nested Cross-Validation (5x3 Folds)
  - Train and optimize 10 classifiers (linear and non-linear)
              |
              v
SECTION 10: Holdout Validation & Calibration
  - Evaluate best models on holdout cohort with bootstrap CIs, calibration, and DCA
              |
              v
SECTION 11: SHAP Model Explainability
  - Map predictions globally and locally via SHAP beeswarm and waterfalls
              |
              v
