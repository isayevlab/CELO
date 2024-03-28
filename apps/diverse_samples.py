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
            df = preprocess_data(space)
            selected_indxs = select_init_diverse(dataset=df.values, n_sample=sample_number)
            projected_data = UMAP(n_neighbors=5, min_dist=0.5,
                                  metric='cosine').fit_transform(df.values)
            projected_data = pd.DataFrame(projected_data, columns=["UMAP 1", "UMAP 2"],
                                          index=df.index)
            projected_data["SELECTED"] = 0
            projected_data["SELECTED"].iloc[selected_indxs] = 1
            projected_data["SELECTED"] = projected_data["SELECTED"].astype(bool)
            fig = px.scatter(projected_data,
                             x="UMAP 1",
                             y="UMAP 2",
                             color="SELECTED",
                             color_discrete_map={False: 'grey', True: "red"})
            st.plotly_chart(fig)
            selected = space[projected_data["SELECTED"] == 1]
            st.dataframe(selected)
