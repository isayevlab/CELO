import pandas as pd
import streamlit as st
from umap import UMAP
import plotly.express as px
from modules.preprocess_data import preprocess_data
from modules.select_diverse_samples import select_init_diverse


def sample_selector():
    st.title("Diverse Sample Selection")

    # User can choose the method to select the space
    selection_method = st.radio("Select the method to provide space data:",
                                options=["Upload CSV file", "Enter Experiment Name"])

    space = None

    if selection_method == "Upload CSV file":
        data_file = st.file_uploader("Upload Space CSV file")
        if data_file is not None:
            space = pd.read_csv(data_file, index_col=0)
    elif selection_method == "Enter Experiment Name":
        experiment_name = st.text_input(label="Experiment ID", placeholder="polymer_space_1")
        if experiment_name is not None and len(experiment_name) > 0:
            space = pd.read_csv(f"./experiments/{experiment_name}/space.csv", index_col=0)

    if space is not None:
        sample_number = st.number_input(label="Desired number of samples", value=20)
        clustering_method = st.selectbox("Clustering Method",
                                         options=["butina", "kmeans", "uniform", "max_min"],
                                         index=2)
        reduction_method = st.selectbox("Dimensionality Reduction Method",
                                        options=[None, "umap", "pca"], index=0)
        n_components = None
        if reduction_method is not None:
            n_components = st.number_input(
                label="Number of Components for Dimensionality Reduction", value=2, min_value=1)

        ignore_columns = [col for col in space.columns if col.startswith("__")]
        space_filtered = space.loc[:, ~space.columns.isin(ignore_columns)]

        # Allow user to fix variables
        with st.expander("Fix Variables"):
            fixed_columns = st.multiselect("Select columns to fix", options=space_filtered.columns)
            fixed_values = {}
            for col in fixed_columns:
                unique_values = space_filtered[col].unique()
                fixed_values[col] = st.selectbox(f"Select value for {col}", options=unique_values)

            if fixed_columns:
                query = " & ".join([f"{col} == @fixed_values['{col}']" for col in fixed_columns])
                space_filtered = space_filtered.query(query)

        df = preprocess_data(space_filtered)
        selected_indxs = select_init_diverse(dataset=df.values, n_sample=sample_number,
                                             method=clustering_method,
                                             reduction_method=reduction_method,
                                             n_components=n_components)
        projected_data = UMAP(n_neighbors=5, min_dist=0.5, metric='cosine',
                              n_components=2).fit_transform(df.values)
        tmp_space = space_filtered.copy()
        tmp_space["Dim 1"] = projected_data[:, 0]
        tmp_space["Dim 2"] = projected_data[:, 1]

        tmp_space["SELECTED"] = 0
        tmp_space["SELECTED"].iloc[selected_indxs] = 1
        tmp_space["SELECTED"] = tmp_space["SELECTED"].astype(bool)
        fig = px.scatter(tmp_space,
                         x="Dim 1",
                         y="Dim 2",
                         color="SELECTED",
                         color_discrete_map={False: 'grey', True: "red"},
                         hover_data=list(tmp_space.keys()))
        st.plotly_chart(fig)
        selected = space_filtered.iloc[selected_indxs]

        # Include ignored columns in the final output
        ignored_columns_data = space.loc[selected.index, ignore_columns]
        final_output = pd.concat([selected, ignored_columns_data], axis=1)

        st.dataframe(final_output)


if __name__ == "__main__":
    sample_selector()
