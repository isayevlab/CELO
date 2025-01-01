import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import numpy as np
from scipy.stats import entropy
import streamlit as st



def compute_uncertainty(pred_proba):
    pred_proba = np.array(pred_proba)
    uncertainties = entropy(pred_proba, axis=1)
    return uncertainties


custom_hyperparameters = get_hyperparameter_config('default')

short2long = {
    "RF": "RandomForest",
    "XGB": "XGBoost",
    "XT": "ExtraTrees",
    "CAT": "CatBoost",
    "KNN": "KNeighbors",
    "FASTAI": "NeuralNetFastAI",
    "NN_TORCH": "NeuralNetTorch",
    "GBM": "LightGBM"
}


def get_hyperparameters(name):
    model_type = None
    for k, v in short2long.items():
        if name.startswith(v):
            model_type = k
            break
    assert model_type is not None
    return {model_type: custom_hyperparameters[model_type]}


def run_autogluon(experiment_name, rewards, ensemble_size, st_bar=None, evaluation_method="random_split", hyperparameters=None):
    # Load data
    space = pd.read_csv(f"./experiments/{experiment_name}/space.csv", index_col=0)
    labeled_data = pd.read_csv(f"./experiments/{experiment_name}/labeled_samples.csv")
    features = list(space.columns)
    X = labeled_data.loc[:, features]

    predictor = TabularPredictor(label="reward", path="ag_data/tmp")

    # Determine task type based on rewards
    if len(rewards) > 1:
        y = labeled_data[rewards].mean(axis=1)  # Multi-reward regression
        task_type = "regression"
    else:
        y = labeled_data[rewards[0]]
        task_type = predictor._learner.infer_problem_type(y=y, silent=True)

    X["reward"] = y
    if st_bar is not None:
        st_bar.progress(0, text="ML model selection")

    predictor = predictor.fit(X, fit_weighted_ensemble=False, hyperparameters=hyperparameters)
    model = predictor.model_best
    best_model_type = model.split('_')[0]  # Extract the model type from the model name
    model_hyperparameters = get_hyperparameters(model)

    # Evaluation
    results = []
    if evaluation_method == "leave_one_out":
        st_bar.progress(0, text="LOOCV in progress")
        loo = LeaveOneOut()

        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            tp = TabularPredictor(label="reward", path="ag_data/tmp")
            tp = tp.fit(X_train, fit_weighted_ensemble=False, hyperparameters=model_hyperparameters)
            prediction = tp.predict(X_test)

            if task_type == "regression":
                results.append(prediction.values[0])
            else:
                proba = tp.predict_proba(X_test)
                uncertainty = compute_uncertainty(proba)
                results.append((prediction.values[0], uncertainty[0]))

        if task_type == "regression":
            predictions = pd.DataFrame(results, columns=["Prediction"])
            metrics = {"MSE": mean_squared_error(y, predictions["Prediction"])}
        else:
            predictions, confidences = zip(*results)
            metrics = {"Accuracy": accuracy_score(y, predictions),
                       "F1 Score": f1_score(y, predictions, average="weighted")}

    elif evaluation_method == "random_split":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        tp = TabularPredictor(label="reward", path="ag_data/tmp")
        tp = tp.fit(X_train, fit_weighted_ensemble=False, hyperparameters=model_hyperparameters)
        predictions = tp.predict(X_test)

        if task_type == "regression":
            metrics = {"MSE": mean_squared_error(y_test, predictions)}
        else:
            metrics = {"Accuracy": accuracy_score(y_test, predictions),
                       "F1 Score": f1_score(y_test, predictions, average="weighted")}

    else:
        raise ValueError("Invalid evaluation_method. Choose 'leave_one_out' or 'random_split'.")

    # Final Prediction
    if task_type == "regression":
        ensemble_predictions = []
        for _ in range(ensemble_size):
            tp = TabularPredictor(label="reward", path="ag_data/tmp")
            tp = tp.fit(X_train.sample(frac=1, replace=True),
                        fit_weighted_ensemble=False,
                        hyperparameters=model_hyperparameters)
            ensemble_predictions.append(tp.predict(space[features]))

        # Calculate mean and standard deviation across ensemble
        ensemble_predictions = pd.concat(ensemble_predictions, axis=1)
        mean_predictions = ensemble_predictions.mean(axis=1)
        std_predictions = ensemble_predictions.std(axis=1)
        return mean_predictions, std_predictions, metrics, best_model_type

    else:  # Classification
        ensemble_probas = []
        for _ in range(ensemble_size):
            tp = TabularPredictor(label="reward", path="ag_data/tmp")
            tp = tp.fit(X_train.sample(frac=1, replace=True),
                        fit_weighted_ensemble=False,
                        hyperparameters=model_hyperparameters)
            ensemble_probas.append(tp.predict_proba(space[features]))

        # Average the probabilities for ensemble prediction
        mean_proba = pd.concat(ensemble_probas).groupby(level=0).mean()
        predictions = mean_proba.idxmax(axis=1)  # Get the predicted classes
        uncertainty = compute_uncertainty(mean_proba)  # Compute uncertainty using entropy
        return predictions, uncertainty, metrics, best_model_type
