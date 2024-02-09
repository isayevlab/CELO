import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from lightning.app.components import ServeStreamlit
from umap import UMAP

from modules.preprocess_data import preprocess_data
from modules.select_diverse_samples import select_init_diverse


class DiverseSamplesSelection(ServeStreamlit):
    def render(self):
        data_file = st.file_uploader("CSV data file")
        configuration_file = st.file_uploader("YAML configuration file")

        if st.button("Submit", type="primary"):
            if data_file:
                config = {}
                if configuration_file is not None:
                    config = yaml.load(configuration_file, Loader=yaml.FullLoader)
                index_col = config.get('preprocessing', {}).get('index_col', None)
                features = config.get('preprocessing', {}).get('features', None)
                scaling_columns = config.get('preprocessing', {}).get('scaling_columns', None)
                df = pd.read_csv(data_file, index_col=index_col)
                if features is not None:
                    df = df.loc[:, features]
                df = preprocess_data(df, scaling_columns=scaling_columns)
                selected_indxs = select_init_diverse(dataset=df.values, n_sample=30)
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
                st.download_button('Download selected samples CSV', projected_data.to_csv(),
                                   'selected_samples.csv', type="primary")
