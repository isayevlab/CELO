import os.path

import pandas as pd
import streamlit as st
import plotly.express as px


from lightning.app.components import ServeStreamlit

from modules.ml_models.run_rl import run_rl


class MLModelSelection(ServeStreamlit):
    def render(self):
        experiment_name = st.text_input(label="Experiment ID", placeholder="polymer_space_1")

        if experiment_name is not None:
            space_path = f"experiments/{experiment_name}/space.csv"
            labeled_path = f"experiments/{experiment_name}/labeled_samples.csv"
            if (os.path.exists(f"experiments/{experiment_name}")
                and os.path.exists(f"experiments/{experiment_name}/space.csv")
                and os.path.exists(f"experiments/{experiment_name}/labeled_samples.csv")):

                space = pd.read_csv(space_path, index_col=0)
                labeled_data = pd.read_csv(labeled_path)

                features = list(space.columns)
                labeled_features = list(labeled_data.columns)

                assert set(features) & set(labeled_features) == set(features)

                rewards = st.multiselect(
                    'Select components of the target',
                    list(set(labeled_features) - set(features)))

                type = st.radio(
                    "Type of ML model",
                    ["RL", "Classic ML"],
                    captions=["Reinforcement Learning", "Classical Machine Learning"])
                ensemble_size = st.number_input(label="Desired ensemble size", value=12)
                if st.button("Build ML model", type="primary"):
                    mean, std = run_rl(experiment_name, rewards, ensemble_size)

                    space["Uncertainty"] = std
                    space["Reward"] = mean

                    fig = px.scatter(space,
                                     x="Uncertainty",
                                     y="Reward",
                                     hover_data=features)
                    st.plotly_chart(fig)
                    st.dataframe(space)

            else:
                if not (os.path.exists(f"experiments/{experiment_name}") or
                   not os.path.exists(f"experiments/{experiment_name}/labeled_samples.csv")):
                    st.write(f"The space is not created for the experiment - {experiment_name}")
                else:
                    st.write(f"There is no labeled samples for the experiment - {experiment_name}")