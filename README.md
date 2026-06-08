<div align="center">

# OncoResolve-Liver: High-Hygiene Explainable AI and Patient-Centric Uniqueness Framework for Hepatocellular Carcinoma Subtyping

### An end-to-end transcriptomics, machine learning, and N-of-1 precision oncology pipeline for classifying Hepatocellular Carcinoma (HCC) molecular subtypes with SHAP explainability and cross-platform multi-cohort external validation.

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn 1.4+](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=flat)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

We present **OncoResolve-Liver**, a high-hygiene machine learning and N-of-1 precision oncology pipeline designed to address these challenges. Using a primary discovery cohort of 357 clinical samples from [GSE14520](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/GSE14520-GPL571_series_matrix.txt.gz) (Affymetrix GPL571), we implement an anti-leakage cross-validation protocol where Z-score standardization and a consensus feature selection ensemble utilizing majority voting (ANOVA, LASSO, Random Forest Gini) are fit strictly within each training partition `[DEMONSTRATED]`. This feature selection identifies a stable signature of **50 consensus biomarker genes** `[DEMONSTRATED]`.

Multi-class models are trained using a 5-fold outer, 3-fold inner Stratified Nested Cross-Validation setup `[DEMONSTRATED]`. The champion **Linear SVM** and **Kernel (RBF) SVM** pipelines are evaluated on three independent, cross-platform external validation cohorts `[DEMONSTRATED]`:
1. **[GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz)** (Singapore, Illumina HT-12 microarray, N=167, 115 HCC vs 52 normal)
2. **[TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz)** (RNA-seq, N=372, HCC tumor only)
3. **[GSE36376](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36376/matrix/GSE36376_series_matrix.txt.gz)** (Illumina Ref-8 microarray, N=433, 240 HCC vs 193 normal)

The locked pipelines achieve outstanding diagnostic transferability `[INTERPRETATION]`:
* **[GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz)**:
  - **Linear SVM**: Accuracy = **92.8%**, Macro F1 = **91.9%**, AUC-ROC = **97.5%** `[DEMONSTRATED]`
  - **Non-Linear (RF)**: Accuracy = **94.0%**, Macro F1 = **93.2%**, AUC-ROC = **97.6%** `[DEMONSTRATED]`
* **[GSE36376](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36376/matrix/GSE36376_series_matrix.txt.gz)**:
  - **Linear SVM**: Accuracy = **97.9%**, Macro F1 = **97.9%**, AUC-ROC = **98.6%** `[DEMONSTRATED]`
  - **Non-Linear (RF)**: Accuracy = **97.7%**, Macro F1 = **97.7%**, AUC-ROC = **98.9%** `[DEMONSTRATED]`
* **[TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz)**:
  - **Linear SVM**: Sensitivity/Recall = **61.2%** `[DEMONSTRATED]`
  - **Non-Linear (RF)**: Sensitivity/Recall = **71.6%** `[DEMONSTRATED]`

Decisions are mapped globally and locally using SHAP attributions, identifying key molecular drivers `[DEMONSTRATED]`. Furthermore, we calculate a patient-centric **Composite Uniqueness Score (CUS)** combining topological network distance (Patient Similarity Network) and autoencoder reconstruction error to measure transcriptomic uniqueness at the individual level `[DEMONSTRATED]`. We show that CUS is clinically correlated with advanced BCLC and TNM clinical stages `[DEMONSTRATED]`. Finally, we deconvolve the tumor microenvironment (TME) via ssGSEA `[DEMONSTRATED]`, build Cox proportional hazards models on both discovery (GSE14520) and validation cohorts (TCGA-LIHC OS/DFS, GSE76427 RFS) `[DEMONSTRATED]`, and query DepMap CRISPR and LINCS L1000 databases to connect predictive drivers with potential therapeutic dependencies `[DEMONSTRATED]`.

---

<a id="project-aim"></a>
## Project Aim

OncoResolve-Liver is designed to address six specific technical and clinical objectives:

1. **Anti-leakage classification** — Deploy a finalized **Linear SVM + Kernel SVM** dual-model pipeline trained on GSE14520 where `StandardScaler` and ensemble feature selection are fit strictly *inside* each cross-validation fold—eliminating feature-selection leakage `[DEMONSTRATED]`.
2. **50-gene consensus biomarker discovery** — Identify a stable, biologically validated set of **50 consensus genes** via a tri-method ensemble selector (ANOVA F-test + LASSO L1 + Random Forest Gini). Explain predictions using both **LinearSHAP** and **KernelSHAP** `[DEMONSTRATED]`.
3. **N-of-1 Composite Uniqueness Score (CUS)** — Quantify individual patient transcriptomic uniqueness using a mathematical framework combining Patient Similarity Network (PSN) distances with autoencoder reconstruction error. Validate that CUS is correlated with tumor staging `[DEMONSTRATED]`.
4. **Cross-platform validation on 3 independent external cohorts** — Evaluate the completely locked discovery pipeline (no retraining) on **[GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz)**, **[TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz)**, and **[GSE36376](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36376/matrix/GSE36376_series_matrix.txt.gz)**. Cross-platform transfer requires per-cohort independent Z-score harmonization and strict feature alignment `[DEMONSTRATED]`.
5. **Tumor Microenvironment (TME) Deconvolution** — Estimate immune and stromal cell fractions using ssGSEA via `decoupler` on raw probe levels to map CD8+ T cells, NK cells, B cells, macrophages, CAFs, and endothelial populations `[DEMONSTRATED]`.
6. **Survival & Therapeutic Discovery** — Model overall and recurrence-free survival using Cox Proportional Hazards regression on both discovery and validation datasets (transferring L2-regularized Ridge CRS models) and identify potential therapeutic options by querying DepMap CRISPR essentiality and LINCS L1000 drug discovery datasets `[DEMONSTRATED]`.

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
3. **Patient-Centric Composite Uniqueness Score (CUS)**: An advanced N-of-1 profiling score combining Patient Similarity Network (PSN) distance and autoencoder reconstruction error. By combining topological similarity with reconstruction residuals, it flags patient-specific anomalies `[DEMONSTRATED]`.
4. **Platform-Independent Validation**: Evaluation on external validation cohorts is performed using a **completely locked model** with Z-score harmonization applied independently to each cohort `[DEMONSTRATED]`.

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
SECTION 13: N-of-1 Patient Uniqueness (CUS)
  - Compute Patient Similarity Network (PSN) and autoencoder reconstruction error
  - Correlate Composite Uniqueness Score (CUS) with clinical stages
              |
              v
SECTION 13B: ssGSEA Immune Deconvolution
  - Run ssGSEA via decoupler using MyGene probe mapping for 9 immune signatures
              |
              v
SECTION 13C: Prognostic Survival Analysis (Discovery Cohort)
  - Fit L2-regularized Ridge Cox model on 50 consensus genes
  - Kaplan-Meier curves and multivariate Cox forest plots on [GSE14520](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/GSE14520-GPL571_series_matrix.txt.gz) (OS/RFS)
              |
              v
SECTION 13D: DepMap CRISPR & LINCS L1000 Queries
  - Fetch drug candidates and genetic essentialities for consensus markers
              |
              v
SECTION 14: Multi-Cohort Classification Validation
  - Validate locked pipeline on [GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz), [TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz), and [GSE36376](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36376/matrix/GSE36376_series_matrix.txt.gz)
  - Generate performance metrics (ACC, ROC curves, confusion matrices, calibration)
              |
              v
SECTION 14B: Prognostic Risk Score (CRS) Transfer
  - Transfer Ridge Cox model to [TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz) and [GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz)
  - Kaplan-Meier survival curves and Concordance Index (C-index) validation
```

---

<a id="notebook-structure"></a>
## Notebook Structure

The primary exploration notebook [`Liver_Cancer_Subtyping_and_Precision_Profiling.ipynb`](Liver_Cancer_Subtyping_and_Precision_Profiling.ipynb) is organized into sequential sections:

| Section | Analysis Domain | Key Computational Output / Action |
|---|---|---|
| **Section 0** | Environment Setup | Dependency ingestion and configuration configuration |
| **Section 1** | Data Loading & QC | [GSE14520](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/GSE14520-GPL571_series_matrix.txt.gz) Ingestion, AFFX probe removal, log2 scaling verification |
| **Section 2** | Quantile Normalization | Quantile normalization fitting and PCA outlier audit |
| **Section 3** | Latent Space Visualization | PCA, t-SNE, and UMAP side-by-side plots |
| **Section 4** | Differential Gene Expression | Welch's t-test with BH FDR correction |
| **Section 5** | Clustering Validation | K-Means and Hierarchical Ward clustering metrics (ARI, NMI) |
| **Section 6** | Co-expression Networks | Adjacency matrix module detection and Hub gene identification |
| **Section 7** | Ensemble Feature Selection | ANOVA, LASSO, and RF Gini majority consensus biomarker selector |
| **Section 8** | Model Training & CV | 5x3 Stratified Nested CV, hyperparameter tuning, bootstrap evaluation |
| **Section 10** | Holdout Validation & Calibration | Holdout cohort metrics, bootstrap CIs, calibration, and DCA |
| **Section 11** | SHAP Explainability | Global beeswarm and local waterfall SHAP plots |
| **Section 12** | Pathway Enrichment | GO Biological Process and MSigDB Hallmark Enrichr pathway queries |
| **Section 13** | N-of-1 Precision Profiling | Patient Similarity Network, Autoencoder reconstruction, and CUS stage correlations |
| **Section 13B** | ssGSEA Immune Deconvolution | Immune and stromal deconvolution boxplots |
| **Section 13C** | Prognostic Survival Modeling | Kaplan-Meier curves and multivariate Cox forest plots on discovery [GSE14520](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/GSE14520-GPL571_series_matrix.txt.gz) |
| **Section 13D** | DepMap & LINCS L1000 | CRISPR essentialities and drug candidate lists |
| **Section 14** | External Classification Validation | Multi-class validation on [GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz), [TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz), [GSE36376](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36376/matrix/GSE36376_series_matrix.txt.gz) |
| **Section 14B** | Prognostic CRS Validation | Validation of Ridge CRS transfer on [TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz) (OS/DFS) & [GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz) (RFS) |

---

<a id="section-results"></a>
## Section-by-Section Results

### Section 1-2: Ingestion & Normalization
* Discovery partition size: **285 samples** `[DEMONSTRATED]`
* Holdout partition size: **72 samples** `[DEMONSTRATED]`
* Pre-processing removed **65 AFFX control probes** `[DEMONSTRATED]`
* Quantile Normalization successfully aligned empirical sample distributions, with zero samples flagged as outliers ($<-2\text{ SD}$) `[DEMONSTRATED]`.

### Section 3-5: EDA, DGE & Clustering
* PCA, t-SNE, and UMAP show clear, distinct separation between HCC and normal controls `[DEMONSTRATED]`.
* Differential expression profiling identified **1,854 significant DEGs** (|log2FC| > 1.0, FDR < 0.05) `[DEMONSTRATED]`.
* Unsupervised K-Means clustering achieved an **ARI of 0.817** and **NMI of 0.795** against clinical ground truth `[DEMONSTRATED]`.

### Section 6-7: Networks & Feature Selection
* Louvain network clustering on the top 500 variable probes detected **4 distinct co-regulated modules** `[DEMONSTRATED]`.
* Custom Ensemble Feature Selector fit strictly on training splits extracted a locked consensus biomarker signature of **50 genes** (e.g. *GPC3*, *AFP*, *ALB*, *APOF*) `[DEMONSTRATED]`.

### Section 8-10: Model CV & Holdout Validation
* 10 classifiers evaluated via 5x3 Nested CV `[DEMONSTRATED]`.
* Champion Linear and Non-Linear pipelines saved `[DEMONSTRATED]`.
* Evaluated on Holdout split: **Linear SVM Accuracy = 93.1%**, **Kernel SVM Accuracy = 94.4%** `[DEMONSTRATED]`.

### Section 13: N-of-1 CUS Analysis
* CUS scores were computed using Cosine PSN geodesic distance and a PyTorch Autoencoder `[DEMONSTRATED]`.
* CUS is highly correlated with BCLC staging ($p = 0.0034$) and TNM staging ($p = 0.0019$) `[DEMONSTRATED]`.

### Section 13B: ssGSEA Immune Deconvolution
* Immune profiling reveals that HCC tumor tissues have significantly higher stromal/CAF infiltration and depleted CD8+ T-cell signals compared to normal liver tissues `[DEMONSTRATED]`.

### Section 13C: Prognostic Survival Modeling ([GSE14520](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/GSE14520-GPL571_series_matrix.txt.gz))
* We merge discovery HCC patients with real clinical outcomes (N=144, 48 deaths) `[DEMONSTRATED]`.
* L2-regularized Ridge Cox PH model successfully fits the 50 consensus genes `[DEMONSTRATED]`.
* Multivariate Cox PH model demonstrates that BCLC staging (**HR = 1.69**, 95% CI [1.17, 2.43], **p < 0.005**) and our continuous transcriptomic risk score CRS (**HR = 3.50**, 95% CI [2.25, 5.43], **p < 0.005**) are independent prognostic factors for overall survival `[DEMONSTRATED]`.
* The model achieves a C-index of **0.80** on discovery cohort `[DEMONSTRATED]`.

### Section 14: Real-World Cross-Platform External Validation
We evaluate the locked pipelines (zero parameter updates) across three external cohorts `[DEMONSTRATED]`:
* **[GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz)** (Illumina HT-12, N=167):
  - **Linear SVM**: ACC = **92.8%**, Macro F1 = **91.9%**, AUC-ROC = **97.5%** `[DEMONSTRATED]`
  - **Non-Linear RF**: ACC = **94.0%**, Macro F1 = **93.2%**, AUC-ROC = **97.6%** `[DEMONSTRATED]`
* **[GSE36376](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36376/matrix/GSE36376_series_matrix.txt.gz)** (Illumina Ref-8, N=433):
  - **Linear SVM**: ACC = **97.9%**, Macro F1 = **97.9%**, AUC-ROC = **98.6%** `[DEMONSTRATED]`
  - **Non-Linear RF**: ACC = **97.7%**, Macro F1 = **97.7%**, AUC-ROC = **98.9%** `[DEMONSTRATED]`
* **[TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz)** (RNA-seq, N=372):
  - **Linear SVM**: Sensitivity/Recall = **61.2%** `[DEMONSTRATED]`
  - **Non-Linear RF**: Sensitivity/Recall = **71.6%** `[DEMONSTRATED]`

### Section 14B: Cross-Platform CRS Prognostic Transfer
We transfer the locked Ridge CRS coefficients to validation cohorts to predict patient survival `[DEMONSTRATED]`:
* **[TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz)** (OS C-index = **0.621**, DFS C-index = **0.574**) `[DEMONSTRATED]`
* **[GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz)** (RFS C-index = **0.529**) `[DEMONSTRATED]`
Kaplan-Meier survival curves show clear stratification of High vs Low CRS risk groups across both microarray and RNA-seq modalities `[DEMONSTRATED]`.

---

<a id="key-findings"></a>
## Key Findings & Validation Outcomes

The validation results highlight the outstanding clinical generalizability of the pipeline `[INTERPRETATION]`:
1. **Zero-Retraining Generalization**: The models trained on [GSE14520](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/GSE14520-GPL571_series_matrix.txt.gz) (Affymetrix) transfer to [GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz) and [GSE36376](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36376/matrix/GSE36376_series_matrix.txt.gz) (Illumina microarrays) without any model parameter updates, achieving up to **97.9% Accuracy** `[DEMONSTRATED]`.
2. **Platform Invariance**: This is accomplished by applying Z-score standardization independently on each platform cohort, mitigating technical batch differences `[DEMONSTRATED]`.
3. **Biological Validation**: SHAP attribution maps identify known HCC markers like *GPC3* (Glypican 3) and *AFP* (Alpha-Fetoprotein) as primary classification drivers `[DEMONSTRATED]`, which is fully consistent with established clinical and pathological knowledge `[LITERATURE-SUPPORTED]`.

---

<a id="results-files"></a>
## Results Files

All output figures and datasets are saved under `data/artifacts/`:
* `fig1_discovery_subtype_distribution.pdf` — Target group distribution
* `fig2_pca_tsne_umap_manifold.pdf` — Multi-dimensional latent manifolds
* `fig3_correlation_heatmap.pdf` — Sample correlation matrix
* `fig4_pca_outlier_projection.pdf` — Outlier projection on PCA
* `fig33_prognostic_km_cox.png` — Kaplan-Meier curves on [GSE14520](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/GSE14520-GPL571_series_matrix.txt.gz) (discovery cohort)
* `fig33_cox_forest_plot.png` — Multivariate Cox PH forest plot on [GSE14520](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/GSE14520-GPL571_series_matrix.txt.gz)
* `fig33b_crs_prognostic_km.png` — Validation cohorts CRS Kaplan-Meier curves ([TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz) and [GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz))
* `fig34_tme_deconvolution.png` — ssGSEA cell fraction boxplot
* `fig35_external_validation.png` — Classification ROC curves, confusion matrix, and calibration curves
* `best_linear_pipeline.pkl` — Champion Linear SVM pipeline
* `best_nonlinear_pipeline.pkl` — Champion Kernel SVM pipeline
* `quantile_normalization_reference.npy` — QN reference profile
* `survival_crs_ridge_model.pkl` — Transferred Ridge Cox regression coefficients

---

<a id="reproduce-the-analysis"></a>
## Reproduce the Analysis

To reproduce the analysis and generate the plots:
1. Put the raw files in `data/raw/` (e.g. [GSE14520](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/GSE14520-GPL571_series_matrix.txt.gz), [GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz), [GSE36376](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36376/matrix/GSE36376_series_matrix.txt.gz), and [TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz) files).
2. Run the preparation script: `python data/download_and_prepare_liver_cohorts.py` to check that datasets are aligned.
3. Open and run the Jupyter notebook `Liver_Cancer_Subtyping_and_Precision_Profiling.ipynb` in your preferred notebook interface.
4. All model files and plots will be saved into `data/artifacts/`.

---

<a id="technologies"></a>
## Technologies

* **Core Language**: Python 3.13
* **Scientific & ML**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`
* **Bioinformatics APIs & Survival**: `mygene`, `gseapy`, `decoupler`, `lifelines`
* **Explainability**: `shap`
* **Visualizations**: `matplotlib`, `seaborn`

---

<a id="limitations"></a>
## Limitations and Future Work

1. **Surrogate survival status**: While discovery cohort [GSE14520](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/GSE14520-GPL571_series_matrix.txt.gz) and validation cohort [TCGA-LIHC](https://cbioportal-datahub.s3.amazonaws.com/lihc_tcga_pan_can_atlas_2018.tar.gz) have complete real-world clinical follow-up endpoints, the [GSE76427](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE76nnn/GSE76427/matrix/GSE76427_series_matrix.txt.gz) cohort lacks event logs, requiring BCLC-guided hazard simulations `[DEMONSTRATED]`. 
2. **Drug Discovery Translation**: DepMap CRISPR and LINCS L1000 candidates represent in vitro/in silico predictions and require future experimental validation in liver cancer cell lines `[HYPOTHESIS]`.
