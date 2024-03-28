import os

import pandas as pd
import plotly.express as px
import streamlit as st
from lightning.app.components import ServeStreamlit
from sklearn.cluster import KMeans
from umap import UMAP

from modules.preprocess_data import preprocess_data


class ExplorationExplotation(ServeStreamlit):
    def render(self):
        experiment_name = st.text_input(label="Experiment ID", placeholder="polymer_space_1")
        sample_number = st.number_input(label="Desired number of samples", value=20)
        fraction = st.slider("Fraction of space used for search", min_value=0.01, max_value=1.0,
                             step=0.01)
        n_clusters = st.slider("Number of clusters for diverse selection", min_value=1,
                               max_value=sample_number,
                               step=1)

        if experiment_name is not None:
            space_path = f"experiments/{experiment_name}/space.csv"
            predictions_path = f"experiments/{experiment_name}/predictions.csv"

            if os.path.exists(space_path) and os.path.exists(predictions_path):
                space = pd.read_csv(space_path, index_col=0)
                features = list(space.columns)
                predictions = pd.read_csv(predictions_path, index_col=0)
                space["Reward"] = predictions["Reward"]
                space["Uncertainty"] = predictions["Uncertainty"]
                threshold = predictions["Reward"].quantile(1 - fraction)

                reduced_space = space[predictions["Reward"] >= threshold]

                processed_space = preprocess_data(reduced_space.loc[:, features])

                clusters = KMeans(n_clusters=n_clusters).fit_predict(processed_space)

                reduced_space["clusters"] = clusters

                points_per_cluster = []

                for i in range(n_clusters):
                    points_per_cluster.append(
                        sample_number // n_clusters + (i < sample_number % n_clusters))
                assert sum(points_per_cluster) == sample_number
                selected = []
                for cluster in range(n_clusters):
                    cluster_space = reduced_space[reduced_space["clusters"] == cluster]
                    selected.append(cluster_space.nlargest(points_per_cluster[cluster], "Reward"))

                selected = pd.concat(selected)
                selected.sort_values(by="Reward", inplace=True, ascending=False)

                space["Color"] = "Did not explored"
                space["Color"][reduced_space.index] = "Explored"
                space["Color"][selected.index] = "Selected"

                fig = px.scatter(space,
                                 x="Uncertainty",
                                 y="Reward",
                                 color="Color",
                                 color_discrete_map={"Did not explored": 'grey',
                                                     "Explored": 'blue',
                                                     "Selected": 'red'},
                                 hover_data=features)
                st.plotly_chart(fig)

                projected_data = UMAP(n_neighbors=5, min_dist=0.5,
                                      metric='cosine').fit_transform(processed_space.values)

                reduced_space["UMAP 1"] = projected_data[:, 0]
                reduced_space["UMAP 2"] = projected_data[:, 1]

                reduced_space["Color"] = clusters
                reduced_space["Color"] = "Cluster " + reduced_space["Color"].astype(str)
                reduced_space["Color"][selected.index] = "Selected"

                fig = px.scatter(reduced_space,
                                 x="UMAP 1",
                                 y="UMAP 2",
                                 color="Color",
                                 hover_data=features)
                st.plotly_chart(fig)
                st.dataframe(selected)
