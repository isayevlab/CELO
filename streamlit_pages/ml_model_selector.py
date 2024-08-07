import os.path
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.ml_models.auto_glueon import run_autogluon
from modules.ml_models.run_rl import run_rl

def ml_model_selector():
    st.title("ML Model Selector")

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

            model_type = st.radio(
                "Type of ML model",
                ["RL", "Classic ML"],
                captions=["Reinforcement Learning", "Classical Machine Learning"])
            ensemble_size = st.number_input(label="Desired ensemble size", value=12)
            if st.button("Build ML model", type="primary"):
                st_bar = st.progress(0, text="ML model training is starting")
                if model_type == "RL":
                    mean, std = run_rl(experiment_name, rewards, ensemble_size, st_bar)
                else:
                    mean, std = run_autogluon(experiment_name, rewards, ensemble_size, st_bar)
                st_bar.empty()

                predictions = pd.DataFrame(index=space.index)
                predictions["Reward"] = mean
                predictions["Uncertainty"] = std
                predictions_path = f"experiments/{experiment_name}/predictions.csv"
                predictions.to_csv(predictions_path)

                space["Reward"] = mean
                space["Uncertainty"] = std
                fig = px.scatter(space,
                                 x="Uncertainty",
                                 y="Reward",
                                 hover_data=features)
                st.plotly_chart(fig)
                st.dataframe(space)

        else:
            if not os.path.exists(f"experiments/{experiment_name}") or not os.path.exists(f"experiments/{experiment_name}/labeled_samples.csv"):
                st.write(f"The space is not created for the experiment - {experiment_name}")
            else:
                st.write(f"There is no labeled samples for the experiment - {experiment_name}")
