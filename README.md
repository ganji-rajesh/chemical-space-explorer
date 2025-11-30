# 🧪 Chemical Space Explorer

A portfolio-ready web application for visualizing and exploring chemical space using dimensionality reduction and clustering. Combines data science (UMAP, K-Means) with computational chemistry (RDKit, Morgan fingerprints).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![RDKit](https://img.shields.io/badge/RDKit-2023.3+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)

## 🎯 Features

### Core Functionality
- **Interactive Chemical Space Visualization**: Explore molecules in 2D UMAP space
- **Multiple Input Formats**: CSV, SDF, SMILES, MOL, MOL2
- **Automatic Clustering**: K-Means clustering of similar molecules
- **Scaffold Analysis**: Maximum Common Substructure (MCS) computation
- **Built-in Reference Dataset**: ChEMBL approved drugs with therapeutic labels

### Data Science
- **Morgan Fingerprints**: 2048-bit circular fingerprints
- **UMAP**: Dimensionality reduction preserving local structure
- **K-Means**: Configurable clustering (2-50 clusters)
- **Properties**: LogP, MW, TPSA, HBA, HBD, rotatable bonds, aromatic rings

### Chemistry
- **Structure Display**: Interactive 2D molecule rendering
- **Cluster Representatives**: Medoid-based selection
- **Common Scaffolds**: Shared substructure identification
- **Therapeutic Labels**: ATC classification, target families (built-in dataset)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/chemical-space-explorer.git
cd chemical-space-explorer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Build ChEMBL Dataset (Optional)

```bash
python scripts/build_chembl_snapshot.py
```

This fetches ~2000 FDA-approved drugs from ChEMBL.

## 📖 Usage

### 1. Load Data

**Option A: Built-in Dataset**
- Select "Built-in ChEMBL Drugs"
- Includes FDA-approved drugs with therapeutic labels

**Option B: Upload Custom**
- CSV with `smiles` column
- SDF, SMILES (.smi/.txt), MOL/MOL2 files
- Max 5000 molecules

### 2. Configure Parameters

**UMAP:**
- **N Neighbors** (5-50): Local vs global structure
- **Min Distance** (0-0.5): Point packing tightness

**Clustering:**
- **Clusters** (2-50): Number of K-Means clusters

**Visualization:**
- **Label Type**: Therapeutic labels (built-in only)
- **Color By**: Cluster or molecular property

### 3. Explore

1. Click "Compute/Recompute"
2. Interact with chemical space plot
3. Select clusters to view:
   - Statistics
   - Representative molecules
   - Common scaffolds
   - Label distributions

## 🏗️ Project Structure

```
chemical-space-explorer/
├── config.yaml              # All hyperparameters
├── requirements.txt         # Dependencies
├── app.py                   # Streamlit application
├── src/
│   ├── featurization.py    # Fingerprints & properties
│   ├── embedding.py        # UMAP
│   ├── clustering.py       # K-Means
│   ├── mcs.py              # Common substructure
│   └── utils.py            # Data loading
├── tests/
│   ├── test_pipeline.py    # Unit tests
│   └── data/
│       └── golden_test.csv
├── scripts/
│   └── build_chembl_snapshot.py
└── data/
    └── chembl_approved_drugs.parquet
```

## ⚙️ Configuration

Edit `config.yaml` to modify:

```yaml
# Fingerprints
fingerprint:
  radius: 2
  n_bits: 2048

# UMAP
umap:
  n_neighbors: 15
  min_dist: 0.1
  metric: "jaccard"
  random_state: 42

# K-Means
kmeans:
  default_n_clusters: 10
  random_state: 42
```

All parameters use fixed random seeds for reproducibility.

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

Tests include:
- SMILES validation
- Fingerprint computation
- UMAP determinism
- K-Means clustering
- Full pipeline integration

## 🔬 Technical Pipeline

```
Input → Validation → Canonicalization
  ↓
Morgan Fingerprints (2048-bit, radius=2)
  ↓
UMAP (2D, Jaccard metric)
  ↓
K-Means Clustering
  ↓
Cluster Analysis (representatives, MCS)
```

## ⚠️ Bias & Limitations

### Approved Drug Bias
Built-in dataset contains only FDA-approved drugs, not representing full chemical space.

### Label Availability
Therapeutic labels only available for built-in dataset. User uploads have computed properties only.

### 2D Fingerprints
Morgan fingerprints don't capture 3D conformation or stereochemistry.

### UMAP Limitations
- 2D projection loses high-dimensional information
- Local structure preservation may distort global relationships
- Different seeds produce different (but valid) projections

### Similarity ≠ Activity
Clusters based on structural similarity, not biological activity. Similar structures may have different targets.

## 📊 Example Workflows

### Find Drug Clusters
1. Load built-in ChEMBL dataset
2. Color by "ATC Level 1" 
3. Identify cardiovascular drug cluster
4. View common scaffold

### Analyze Custom Library
1. Upload CSV with SMILES
2. Compute with default settings
3. Identify diverse clusters
4. Export cluster representatives

### Compare Properties
1. Load dataset
2. Color by "LogP"
3. Identify high/low lipophilicity regions
4. Cross-reference with clusters

## 🤝 Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- **RDKit**: Open-source cheminformatics
- **ChEMBL**: Drug database
- **UMAP**: Dimensionality reduction algorithm
- **Streamlit**: Web framework

## 📧 Contact

Your Name - your.email@example.com

Project Link: https://github.com/yourusername/chemical-space-explorer

---

**Built with** ❤️ **for portfolio and drug discovery**
