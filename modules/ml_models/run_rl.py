from copy import deepcopy

import torch
from lightning.pytorch.cli import LightningCLI

from modules.ml_models.double_q_learning_model import DQNLightning, EnsembledModel
from modules.ml_models.table_tensor_dataset import TableDatamodule


def run_rl(experiment_name, rewards, ensemble_size):
    args = ["--config", "./configs/rl_model.yaml",
            r"--data.space_path", f"./experiments/{experiment_name}/space.csv",
            "--data.labeled_path", f"./experiments/{experiment_name}/labeled_samples.csv",
            "--data.target_names", ", ".join(rewards)
            ]
    models = []
    for _ in range(ensemble_size):
        cli = LightningCLI(DQNLightning, TableDatamodule,
                           save_config_kwargs={"overwrite": True},
                           args=args, run=False)
        cli.trainer.fit(cli.model, cli.datamodule)
        models.append(deepcopy(cli.model.policy))
    model_agency = EnsembledModel(models)

    cli.model.policy = model_agency
    predictions = cli.trainer.predict(cli.model, datamodule=cli.datamodule)

    rewards = [i[0] for i in predictions]
    stds = [i[1] for i in predictions]

    rewards = torch.cat(rewards).cpu().detach().numpy()
    stds = torch.cat(stds).cpu().detach().numpy()

    return rewards, stds


if __name__ == "__main__":
    run_rl("past_experiment",
           ["Strain Reward",
            "Stress Reward",
            "Toughness Reward"])
