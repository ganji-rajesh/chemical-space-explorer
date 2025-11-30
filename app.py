"""Chemical Space Explorer - Streamlit Web Application."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import warnings

from src.utils import load_config, load_molecules
from src.featurization import compute_fingerprints, compute_properties
from src.embedding import compute_umap_embedding
from src.clustering import perform_clustering, get_cluster_representatives, get_cluster_statistics
from src.mcs import compute_mcs, smarts_to_mol
from rdkit import Chem
from rdkit.Chem import Draw

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Chemical Space Explorer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_configuration():
    """Load configuration file."""
    return load_config("config.yaml")


@st.cache_data
def load_builtin_dataset():
    """Load precomputed built-in ChEMBL dataset."""
    builtin_path = Path("data/chembl_approved_drugs.parquet")
    if builtin_path.exists():
        return pd.read_parquet(builtin_path)
    return None


@st.cache_data
def process_dataset(df, config):
    """Process dataset: compute fingerprints, properties, embeddings, and clusters."""
    df = compute_properties(df)
    fingerprints = compute_fingerprints(df, config)
    embedding = compute_umap_embedding(fingerprints, config)
    labels = perform_clustering(embedding, config)
    return df, fingerprints, embedding, labels


def create_scatter_plot(df, embedding, labels, color_by, label_type=None):
    """Create interactive scatter plot of chemical space."""
    plot_df = df.copy()
    plot_df['UMAP_1'] = embedding[:, 0]
    plot_df['UMAP_2'] = embedding[:, 1]
    plot_df['Cluster'] = labels.astype(str)

    hover_data = {
        'smiles': True,
        'name': True,
        'Molecular Weight': ':.2f',
        'LogP': ':.2f',
        'TPSA': ':.2f'
    }

    if label_type and label_type in plot_df.columns:
        hover_data[label_type] = True

    if color_by == 'Cluster':
        fig = px.scatter(
            plot_df, x='UMAP_1', y='UMAP_2', color='Cluster',
            hover_data=hover_data,
            title='Chemical Space - UMAP Projection',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    elif color_by in plot_df.columns:
        fig = px.scatter(
            plot_df, x='UMAP_1', y='UMAP_2', color=color_by,
            hover_data=hover_data,
            title=f'Chemical Space colored by {color_by}',
            color_continuous_scale='Viridis'
        )
    else:
        fig = px.scatter(
            plot_df, x='UMAP_1', y='UMAP_2',
            hover_data=hover_data,
            title='Chemical Space - UMAP Projection'
        )

    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white')))
    fig.update_layout(height=600, hovermode='closest')
    return fig


def display_molecule(smiles, width=200, height=200):
    """Display molecule structure from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol, size=(width, height))
    return None


def main():
    """Main application function."""
    config = load_configuration()

    st.markdown('<div class="main-header">🧪 Chemical Space Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Visualize and explore molecular chemical space using UMAP and clustering</div>',
        unsafe_allow_html=True
    )

    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    with st.sidebar:
        st.header("⚙️ Controls")

        st.subheader("1. Data Source")
        dataset_type = st.radio(
            "Select dataset",
            ["Built-in ChEMBL Drugs", "Upload Custom Dataset"],
            help="Use pre-loaded approved drugs or upload your own molecules"
        )

        df = None

        if dataset_type == "Built-in ChEMBL Drugs":
            builtin_df = load_builtin_dataset()
            if builtin_df is not None:
                df = builtin_df
                st.success(f"✅ Loaded {len(df)} approved drugs")
            else:
                st.warning("Built-in dataset not found. Please use custom upload.")
        else:
            uploaded_file = st.file_uploader(
                "Upload molecule file",
                type=['csv', 'sdf', 'smi', 'txt', 'mol', 'mol2'],
                help="Supported formats: CSV, SDF, SMILES, MOL, MOL2"
            )

            if uploaded_file:
                try:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    df, errors = load_molecules(temp_path, config)

                    if errors:
                        st.warning(f"⚠️ {len(errors)} errors found")
                        if st.checkbox("Show errors"):
                            for error in errors[:10]:
                                st.text(error)

                    st.success(f"✅ Loaded {len(df)} valid molecules")
                    Path(temp_path).unlink()

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.divider()

        st.subheader("2. UMAP Parameters")
        n_neighbors = st.slider("N Neighbors", 5, 50, config['umap']['n_neighbors'])
        min_dist = st.slider("Min Distance", 0.0, 0.5, config['umap']['min_dist'], 0.05)

        st.divider()

        st.subheader("3. Clustering")
        n_clusters = st.slider(
            "Number of Clusters",
            config['kmeans']['min_clusters'],
            config['kmeans']['max_clusters'],
            config['kmeans']['default_n_clusters']
        )

        st.divider()

        st.subheader("4. Visualization")

        has_labels = df is not None and any(
            label in df.columns for label in config['labels']['available']
        )

        if has_labels:
            available_labels = [
                label for label in config['labels']['available'] if label in df.columns
            ]
            label_type = st.selectbox("Label Type", ["None"] + available_labels)
            label_type = None if label_type == "None" else label_type
        else:
            label_type = None
            st.info("Labels only available for built-in dataset")

        color_by = st.selectbox("Color by", ['Cluster'] + config['properties'])

        st.divider()
        compute_button = st.button("🚀 Compute / Recompute", use_container_width=True, type="primary")

    if df is not None:
        if compute_button or st.session_state.processed_data is None:
            with st.spinner("Processing molecules..."):
                config['umap']['n_neighbors'] = n_neighbors
                config['umap']['min_dist'] = min_dist

                df_processed, fingerprints, embedding, labels = process_dataset(df, config)
                labels = perform_clustering(embedding, config, n_clusters=n_clusters)

                st.session_state.processed_data = {
                    'df': df_processed,
                    'fingerprints': fingerprints,
                    'embedding': embedding,
                    'labels': labels,
                    'config': config,
                    'label_type': label_type
                }
                st.success("✅ Processing complete!")

        if st.session_state.processed_data is not None:
            data = st.session_state.processed_data

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Molecules", len(data['df']))
            with col2:
                st.metric("Number of Clusters", len(np.unique(data['labels'])))
            with col3:
                st.metric("Avg MW", f"{data['df']['Molecular Weight'].mean():.1f}")
            with col4:
                st.metric("Avg LogP", f"{data['df']['LogP'].mean():.2f}")

            st.divider()

            fig = create_scatter_plot(
                data['df'], data['embedding'], data['labels'], color_by, label_type
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("📊 Cluster Details")

            unique_clusters = sorted(np.unique(data['labels']))
            selected_cluster = st.selectbox(
                "Select cluster to explore",
                unique_clusters,
                format_func=lambda x: f"Cluster {x}"
            )

            if selected_cluster is not None:
                stats = get_cluster_statistics(
                    selected_cluster, data['df'], data['labels'],
                    label_type if label_type else None
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cluster Size", stats['size'])
                with col2:
                    st.metric("Mean MW", f"{stats['mean_mw']:.1f}")
                with col3:
                    st.metric("Mean LogP", f"{stats['mean_logp']:.2f}")

                rep_indices, medoid_idx = get_cluster_representatives(
                    selected_cluster, data['df'], data['fingerprints'],
                    data['embedding'], data['labels'], data['config']
                )

                if rep_indices:
                    st.subheader("🧬 Representative Molecules")
                    cols = st.columns(min(5, len(rep_indices)))
                    for i, idx in enumerate(rep_indices[:5]):
                        with cols[i]:
                            smiles = data['df'].iloc[idx]['smiles']
                            name = data['df'].iloc[idx].get('name', f'mol_{idx}')
                            img = display_molecule(smiles)
                            if img:
                                st.image(img, caption=name, use_container_width=True)
                            st.caption(f"MW: {data['df'].iloc[idx]['Molecular Weight']:.1f}")

                    st.subheader("🔬 Common Scaffold")
                    with st.spinner("Computing..."):
                        cluster_smiles = data['df'][data['labels'] == selected_cluster]['smiles'].tolist()
                        mcs_smiles = cluster_smiles[:min(10, len(cluster_smiles))]
                        mcs_smarts = compute_mcs(mcs_smiles, timeout=5)

                        if mcs_smarts:
                            mol = smarts_to_mol(mcs_smarts)
                            if mol:
                                scaffold_img = Draw.MolToImage(mol, size=(300, 300))
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    st.image(scaffold_img, caption="Common Substructure")
                            st.code(mcs_smarts, language='text')
                        else:
                            st.info("No significant common substructure found.")

                    if 'label_distribution' in stats and stats['label_distribution']:
                        st.subheader("📈 Label Distribution")
                        label_df = pd.DataFrame(
                            list(stats['label_distribution'].items()),
                            columns=[label_type, 'Count']
                        ).sort_values('Count', ascending=False)
                        st.dataframe(label_df, use_container_width=True)
    else:
        st.info("👆 Please select or upload a dataset from the sidebar.")
        with st.expander("📖 How to use"):
            st.markdown("""
            ### Getting Started
            1. **Choose a dataset**: Built-in ChEMBL drugs or upload your own
            2. **Adjust parameters**: Tune UMAP and clustering settings
            3. **Compute**: Click "Compute/Recompute"
            4. **Explore**: Interact with the plot and examine clusters

            ### Supported Formats
            - **CSV**: Must have `smiles` column
            - **SDF**: Structure-data file
            - **SMILES**: One per line
            - **MOL/MOL2**: Single molecule
            """)


if __name__ == "__main__":
    main()
