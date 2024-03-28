from copy import deepcopy

import torch
import pandas as pd
from lightning.pytorch.cli import LightningCLI

from modules.ml_models.double_q_learning_model import DQNLightning, EnsembledModel
from modules.ml_models.table_tensor_dataset import TableDatamodule


def run_rl(experiment_name, rewards, ensemble_size, st_bar=None):
    space = pd.read_csv(f"./experiments/{experiment_name}/space.csv", index_col=0)
    features = list(space.columns)
    args = ["--config", "./configs/rl_model.yaml",
            r"--data.space_path", f"./experiments/{experiment_name}/space.csv",
            "--data.labeled_path", f"./experiments/{experiment_name}/labeled_samples.csv",
            "--data.target_names", ", ".join(rewards),
            "--model.input_dim", f"{len(features)}"
            ]
    models = []
    for i in range(ensemble_size):
        cli = LightningCLI(DQNLightning, TableDatamodule,
                           save_config_kwargs={"overwrite": True},
                           args=args, run=False)
        cli.trainer.fit(cli.model, cli.datamodule)
        models.append(deepcopy(cli.model.target))
        if st_bar is not None:
            st_bar.progress((i+1)/ensemble_size,
                            text=f"The {i+1}/{ensemble_size} models have been trained")
    model_agency = EnsembledModel(models)

    cli.model.target = model_agency
    predictions = cli.trainer.predict(cli.model, datamodule=cli.datamodule)

    rewards = [i[0] for i in predictions]
    stds = [i[1] for i in predictions]

    rewards = torch.cat(rewards).cpu().detach().numpy()
    stds = torch.cat(stds).cpu().detach().numpy()

    return rewards, stds


if __name__ == "__main__":
    run_rl("exp_1",
           ["Strain Reward",
            "Stress Reward",
            "Toughness Reward"],
           ensemble_size=3)
