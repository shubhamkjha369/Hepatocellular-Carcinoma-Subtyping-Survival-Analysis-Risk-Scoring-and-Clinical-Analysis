import os
import sys
import json
import gzip
import shutil
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Workspace directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
TMP_DIR = DATA_DIR / "_tmp"

shutil.rmtree(TMP_DIR, ignore_errors=True)
for d in [RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, TMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CBIO_API = "https://www.cbioportal.org/api"
HEADERS  = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

def get_json(url, params=None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def download_cbio_expression(study_id, profile_id, out_csv, label="dataset"):
    """Download expression matrix from cBioPortal for specific consensus genes using POST fetch."""
    if out_csv.exists():
        print(f"  {label}: already present -- skipping.")
        return

    if consensus_genes is None or len(consensus_genes) == 0:
        print(f"  [{label}] ERROR: No consensus genes loaded! Cannot query fetch API.")
        return

    import mygene
    mg = mygene.MyGeneInfo()
    print(f"  [{label}] Resolving gene symbols to Entrez IDs via MyGene...")
    res = mg.querymany(consensus_genes, scopes='symbol', fields='entrezgene', species='human')
    
    entrez_to_symbol = {}
    entrez_ids = []
    for r in res:
        if 'entrezgene' in r:
            eid = int(r['entrezgene'])
            entrez_to_symbol[eid] = r['query']
            entrez_ids.append(eid)

    print(f"  [{label}] Resolved {len(entrez_ids)} consensus genes.")

    print(f"  [{label}] Fetching expression via POST fetch API (may take 10-30s)...")
    url = f"{CBIO_API}/molecular-profiles/{profile_id}/molecular-data/fetch"
    payload = {
        "sampleListId": f"{study_id}_all",
        "entrezGeneIds": entrez_ids
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/json'})
    
    with urllib.request.urlopen(req, timeout=120) as r:
        records = json.loads(r.read())
    print(f"  [{label}] Records: {len(records):,}")

    df = pd.DataFrame(records)
    df["hugoGeneSymbol"] = df["entrezGeneId"].map(entrez_to_symbol)

    expr = df.pivot_table(index="sampleId", columns="hugoGeneSymbol",
                          values="value", aggfunc="mean")
    expr.index.name = "sample_id"
    expr.columns.name = None
    print(f"  [{label}] Matrix: {expr.shape[0]} samples x {expr.shape[1]} genes")
    expr.to_csv(out_csv)
    print(f"  [{label}] Saved -> {out_csv.name}")

def download_cbio_clinical(study_id, out_csv, label="clinical"):
    """Download merged patient+sample clinical data from cBioPortal."""
    if out_csv.exists():
        print(f"  [{label} clinical]: already present -- skipping.")
        return

    print(f"  [{label}] Fetching clinical data ...")
    samples_raw = get_json(f"{CBIO_API}/studies/{study_id}/samples",
                           {"pageSize": 50000, "pageNumber": 0})
    pat_map = {s["sampleId"]: s["patientId"] for s in samples_raw}

    pat_raw = get_json(f"{CBIO_API}/studies/{study_id}/clinical-data",
                       {"clinicalDataType": "PATIENT", "pageSize": 200000})
    pat_records = {}
    for rec in pat_raw:
        pat_records.setdefault(rec["patientId"], {})[rec["clinicalAttributeId"]] = rec["value"]
    pat_df = pd.DataFrame.from_dict(pat_records, orient="index")
    pat_df.index.name = "patient_id"

    samp_raw = get_json(f"{CBIO_API}/studies/{study_id}/clinical-data",
                        {"clinicalDataType": "SAMPLE", "pageSize": 200000})
    samp_records = {}
    for rec in samp_raw:
        samp_records.setdefault(rec["sampleId"], {})[rec["clinicalAttributeId"]] = rec["value"]
    samp_df = pd.DataFrame.from_dict(samp_records, orient="index")
    samp_df.index.name = "sample_id"
    samp_df["patient_id"] = samp_df.index.map(pat_map)

    merged = samp_df.merge(pat_df, on="patient_id", how="left", suffixes=("_sample","_patient"))
    print(f"  [{label}] Shape: {merged.shape}")
    merged.to_csv(out_csv)
    print(f"  [{label}] Saved -> {out_csv.name}")

def progress_hook(count, block, total):
    if total > 0:
        pct = min(int(count * block * 100 / total), 100)
        mb  = count * block // 1048576
        if count % 200 == 0 or pct == 100:
            print(f"\r    {pct}%  ({mb} MB)", end="", flush=True)

# =============================================================================
# Load consensus genes to only extract necessary features for GSE36376
# =============================================================================
try:
    probe_mapping = joblib.load(ARTIFACTS_DIR / "probe_to_symbol_dict.pkl")
    consensus_genes = list(set(probe_mapping.values()))
    print(f"[INFO] Loaded {len(consensus_genes)} consensus gene symbols for filtering.")
except Exception as e:
    consensus_genes = []
    print(f"[WARN] Could not load consensus genes: {e}. All genes will be parsed.")

# =============================================================================
# 1. GSE14520 CLINICAL SUPPLEMENT DOWNLOAD & PARSE
# =============================================================================
print("\n" + "="*65)
print("COHORT: GSE14520 Clinical Supplement")
print("="*65)
GSE14520_CLIN_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/suppl/GSE14520_Extra_Supplement.txt.gz"
GSE14520_GZ = TMP_DIR / "GSE14520_Extra_Supplement.txt.gz"
GSE14520_OUT = RAW_DIR / "GSE14520_clinical.csv"

if GSE14520_OUT.exists():
    print("  GSE14520 clinical data: already present -- skipping.")
else:
    print("  Downloading extra supplement table...")
    urllib.request.urlretrieve(GSE14520_CLIN_URL, GSE14520_GZ, reporthook=progress_hook)
    print("\n  Parsing clinical metadata table...")
    df_c = pd.read_csv(GSE14520_GZ, sep="\t", compression="gzip")
    df_c.to_csv(GSE14520_OUT, index=False)
    print(f"  Saved -> {GSE14520_OUT.name} ({len(df_c)} rows)")

# =============================================================================
# 2. TCGA-LIHC DOWNLOAD & PARSE (cBioPortal, RNA-seq)
# =============================================================================
print("\n" + "="*65)
print("COHORT: TCGA-LIHC  (RNA-seq RSEM, N=372)")
print("="*65)
TCGA_EXPR_OUT = RAW_DIR / "TCGA_LIHC_expression.csv"
TCGA_CLIN_OUT = RAW_DIR / "TCGA_LIHC_clinical.csv"

try:
    download_cbio_expression(
        "lihc_tcga_pan_can_atlas_2018",
        "lihc_tcga_pan_can_atlas_2018_rna_seq_v2_mrna",
        TCGA_EXPR_OUT,
        label="TCGA-LIHC RNA-seq"
    )
    download_cbio_clinical(
        "lihc_tcga_pan_can_atlas_2018",
        TCGA_CLIN_OUT,
        label="TCGA-LIHC clinical"
    )
except Exception as e:
    print(f"  ERROR downloading TCGA-LIHC: {e}")

# =============================================================================
# 3. GSE36376 DOWNLOAD & PARSE (GEO, Illumina HT-12 microarray, N=480)
# =============================================================================
print("\n" + "="*65)
print("COHORT: GSE36376  (Illumina Microarray, N=480)")
print("="*65)
GSE36376_MATRIX_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36376/matrix/GSE36376_series_matrix.txt.gz"
GSE36376_PLATFORM_URL = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL10nnn/GPL10558/annot/GPL10558.annot.gz"

GSE36376_GZ = TMP_DIR / "GSE36376_matrix.txt.gz"
GPL10558_GZ = TMP_DIR / "GPL10558.annot.gz"
GSE36376_EXPR_OUT = RAW_DIR / "GSE36376_expression.csv"
GSE36376_CLIN_OUT = RAW_DIR / "GSE36376_clinical.csv"

if GSE36376_EXPR_OUT.exists() and GSE36376_CLIN_OUT.exists():
    print("  GSE36376 data: already present -- skipping.")
else:
    # Download Series Matrix
    if not GSE36376_GZ.exists():
        print("  Downloading series matrix (~25 MB) ...")
        urllib.request.urlretrieve(GSE36376_MATRIX_URL, GSE36376_GZ, reporthook=progress_hook)
        print()

    # Download Platform Annotation
    if not GPL10558_GZ.exists():
        print("  Downloading platform annotation GPL10558 (~4 MB) ...")
        urllib.request.urlretrieve(GSE36376_PLATFORM_URL, GPL10558_GZ, reporthook=progress_hook)
        print()

    # Parse Annotation Mapping
    print("  Parsing platform annotation to map Illumina probe ID to Gene Symbol...")
    probe_to_symbol = {}
    with gzip.open(GPL10558_GZ, "rt", encoding="utf-8", errors="ignore") as f:
        in_table = False
        col_idx = 2  # Default to index 2
        for line in f:
            line = line.strip()
            if line.startswith("!platform_table_begin"):
                in_table = True
                # The next line is the header row
                header_line = f.readline().strip()
                headers = [h.strip().lower() for h in header_line.split("\t")]
                if "gene symbol" in headers:
                    col_idx = headers.index("gene symbol")
                    print(f"  [GPL10558] Found 'Gene symbol' at column index {col_idx}")
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table and line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) > col_idx:
                    probe_id = parts[0].strip()
                    symbol = parts[col_idx].strip()
                    probe_to_symbol[probe_id] = symbol

    print(f"  Loaded {len(probe_to_symbol)} probe mappings from GPL10558.")

    # Parse Clinical characteristics
    print("  Parsing clinical characteristics from series matrix ...")
    sample_ids = []
    clinical_data = {}
    with gzip.open(GSE36376_GZ, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [s.strip('"') for s in line.split("\t")[1:]]
                for s_id in sample_ids:
                    clinical_data[s_id] = {}
            elif line.startswith("!Sample_characteristics_ch1"):
                parts = [p.strip('"') for p in line.split("\t")[1:]]
                for idx, part in enumerate(parts):
                    if ":" in part:
                        key, val = part.split(":", 1)
                        key = key.strip().replace(" ", "_").lower()
                        s_id = sample_ids[idx]
                        clinical_data[s_id][key] = val.strip()
            elif line.startswith("!series_matrix_table_begin"):
                break

    clin_df = pd.DataFrame.from_dict(clinical_data, orient="index")
    clin_df.index.name = "sample_id"
    clin_df.to_csv(GSE36376_CLIN_OUT)
    print(f"  Saved -> {GSE36376_CLIN_OUT.name} ({len(clin_df)} samples)")

    # Parse Expression table
    print("  Parsing expression data (filtering for consensus genes)...")
    expr_records = []
    with gzip.open(GSE36376_GZ, "rt", encoding="utf-8", errors="ignore") as f:
        in_table = False
        for line in f:
            line = line.strip()
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table and line:
                parts = line.split("\t")
                probe_id = parts[0].strip('"')
                symbol = probe_to_symbol.get(probe_id, "")
                # If symbol is empty or not in consensus_genes (if filtering is enabled)
                if not symbol:
                    continue
                if consensus_genes is not None and len(consensus_genes) > 0 and symbol not in consensus_genes:
                    continue
                
                vals = [float(v) if v != "null" and v != "" else np.nan for v in parts[1:]]
                expr_records.append([symbol] + vals)

    # Pivot / aggregate expression (take mean if multiple probes map to same symbol)
    header = ["gene_symbol"] + sample_ids
    expr_df = pd.DataFrame(expr_records, columns=header)
    expr_df = expr_df.groupby("gene_symbol").mean().T
    expr_df.index.name = "sample_id"
    
    expr_df.to_csv(GSE36376_EXPR_OUT)
    print(f"  Saved -> {GSE36376_EXPR_OUT.name} ({expr_df.shape[0]} samples x {expr_df.shape[1]} genes)")

# Cleanup temp files
print("\nCleaning up temp files...")
try:
    shutil.rmtree(TMP_DIR)
except Exception:
    pass

print("\n" + "="*65)
print("ALL NEW DATASETS ACQUIRED SUCCESSFULLY")
print("="*65)
