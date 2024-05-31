import pandas as pd
import plotly.express as px
import streamlit as st
from lightning.app.components import ServeStreamlit
from umap import UMAP

from modules.preprocess_data import preprocess_data
from modules.select_diverse_samples import select_init_diverse


class DiverseSamplesSelection(ServeStreamlit):
    def render(self):
        experiment_name = st.text_input(label="Experiment ID", placeholder="polymer_space_1")
        sample_number = st.number_input(label="Desired number of samples", value=20)
        data_file = st.file_uploader("Space CSV file")

        space = None

        if data_file is not None:
            space = pd.read_csv(data_file, index_col=0)

        if experiment_name is not None and len(experiment_name) > 0:
            space = pd.read_csv(f"./experiments/{experiment_name}/space.csv", index_col=0)
        if space is not None:
            ignore_columns = [col for col in space.columns if col.startswith("__")]
            df = preprocess_data(space.loc[:, ~space.columns.isin(ignore_columns)])
            selected_indxs = select_init_diverse(dataset=df.values, n_sample=sample_number)
            projected_data = UMAP(n_neighbors=5, min_dist=0.5,
                                  metric='cosine').fit_transform(df.values)
            tmp_space = space.copy()
            tmp_space["UMAP 1"] = projected_data[:, 0]
            tmp_space["UMAP 2"] = projected_data[:, 1]

            tmp_space["SELECTED"] = 0
            tmp_space["SELECTED"].iloc[selected_indxs] = 1
            tmp_space["SELECTED"] = tmp_space["SELECTED"].astype(bool)
            fig = px.scatter(tmp_space,
                             x="UMAP 1",
                             y="UMAP 2",
                             color="SELECTED",
                             color_discrete_map={False: 'grey', True: "red"},
                             hover_data=list(tmp_space.keys()))
            st.plotly_chart(fig)
            selected = space.iloc[selected_indxs]
            st.dataframe(selected)
