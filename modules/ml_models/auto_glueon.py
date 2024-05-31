import pandas as pd

from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

custom_hyperparameters = get_hyperparameter_config('default')

short2long = {
    "RF": "RandomForest",
    "XGB": "XGBoost",
    "XT": "ExtraTrees",
    "CAT": "CATBoost",
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


def run_autogluon(experiment_name, rewards, ensemble_size, st_bar=None):
    space = pd.read_csv(f"./experiments/{experiment_name}/space.csv", index_col=0)
    labeled_data = pd.read_csv(f"./experiments/{experiment_name}/labeled_samples.csv")
    features = list(space.columns)
    X = labeled_data.loc[:, features]
    y = labeled_data.loc[:, rewards]
    y = y.mean(axis=1)

    X["reward"] = y
    if st_bar is not None:
        st_bar.progress(0, text=f"ML model selection")

    predictor = TabularPredictor(label="reward", path="ag_data/tmp").fit(X,
                                                                         fit_weighted_ensemble=False)

    model = predictor.get_model_best()
    model_hyperparameters = get_hyperparameters(model)

    predictions = []

    for i in range(ensemble_size):
        X = X.sample(frac=1)
        tp = TabularPredictor(label="reward", path="ag_data/tmp")
        tp = tp.fit(X,
                    fit_weighted_ensemble=False,
                    hyperparameters=model_hyperparameters)
        predictions.append(tp.predict(space))
    predictions = pd.concat(predictions, axis=1)
    mean = predictions.mean(axis=1)
    std = predictions.std(axis=1)
    return mean.values, std.values
