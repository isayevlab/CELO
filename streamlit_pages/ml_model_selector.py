import os.path
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.ml_models.auto_glueon import run_autogluon, short2long, custom_hyperparameters


def ml_model_selector():
    st.title("ML Model Selector")

    experiment_name = st.text_input(label="Experiment ID", placeholder="polymer_space_1")

    if experiment_name:
        space_path = f"experiments/{experiment_name}/space.csv"
        labeled_path = f"experiments/{experiment_name}/labeled_samples.csv"

        if os.path.exists(space_path) and os.path.exists(labeled_path):
            space = pd.read_csv(space_path, index_col=0)
            labeled_data = pd.read_csv(labeled_path)

            # Ignore columns starting with '__'
            ignore_columns = [col for col in space.columns if col.startswith("__")]
            space_filtered = space.loc[:, ~space.columns.isin(ignore_columns)].copy()
            labeled_data_filtered = labeled_data.loc[:, ~labeled_data.columns.isin(ignore_columns)]

            features = list(space_filtered.columns)
            labeled_features = list(labeled_data_filtered.columns)

            # Ensure the features in labeled data are a subset of space data features
            assert set(features) & set(labeled_features) == set(features)

            # Select reward components
            rewards = st.multiselect(
                'Select components of the target',
                list(set(labeled_features) - set(features))
            )

            # Select models for training
            selected_models = st.multiselect(
                'Select models to use',
                list(short2long.keys()),
                default=list(short2long.keys())
            )

            # Set ensemble size for model predictions
            ensemble_size = st.number_input(label="Desired ensemble size for regression models",
                                            value=12)

            # Choose evaluation method
            evaluation_method = st.radio(
                "Evaluation Method",
                ["Random Split", "Leave-One-Out"],
                index=0
            )

            # Start model building
            if st.button("Build ML Model", type="primary"):
                st_bar = st.progress(0, text="ML model training is starting")

                # Run AutoGluon model with selected hyperparameters
                c_hyperparameters = {i: custom_hyperparameters[i] for i in selected_models}
                mean, std, metrics, best_model_type = run_autogluon(
                    experiment_name, rewards, ensemble_size, st_bar,
                    evaluation_method="leave_one_out" if evaluation_method == "Leave-One-Out" else "random_split",
                    hyperparameters=c_hyperparameters
                )

                st_bar.empty()

                # Display metrics
                st.subheader("Best Model Type")
                st.write(f"The best model type is: {best_model_type}")
                st.subheader("Model Performance Metrics")
                st.write(metrics)

                # Save predictions
                predictions = pd.DataFrame(index=space.index)
                predictions["Prediction"] = mean
                predictions["Uncertainty"] = std
                predictions_path = f"experiments/{experiment_name}/predictions.csv"
                predictions.to_csv(predictions_path)

                # Visualization
                space_filtered.loc[:, "Prediction"] = mean
                space_filtered.loc[:, "Uncertainty"] = std
                fig = px.scatter(space_filtered,
                                 x="Uncertainty",
                                 y="Prediction",
                                 hover_data=features)
                st.plotly_chart(fig)

                # Include ignored columns in the final output
                ignored_columns_data = space.loc[space_filtered.index, ignore_columns]
                final_output = pd.concat([space_filtered, ignored_columns_data], axis=1)
                st.dataframe(final_output)

        else:
            if not os.path.exists(f"experiments/{experiment_name}"):
                st.write(f"The space is not created for the experiment - {experiment_name}")
            elif not os.path.exists(labeled_path):
                st.write(f"There is no labeled samples for the experiment - {experiment_name}")

if __name__ == "__main__":
    ml_model_selector()
