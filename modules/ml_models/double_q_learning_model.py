import random
from collections.abc import Iterable
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor


class DQN(nn.Module):
    def __init__(self, n_hidden=16,
            state_size=None, action_size=None):
        super(DQN, self).__init__()
        n_total = state_size + action_size
        if not isinstance(n_hidden, Iterable):
            n_hidden = [n_hidden, ]
        fcs = [nn.Linear(n_total, n_hidden[0]), ]
        for prev, next in zip(n_hidden[:-1], n_hidden[1:]):
            fcs.append(nn.Linear(prev, next))
        fcs.append(nn.Linear(n_hidden[-1], 1))
        self.fcs = nn.ModuleList(fcs)

    def forward(self, x):
        for fc in self.fcs:
            x = F.gelu(fc(x))
        return x[..., 0]


class EnsembledModel(nn.Module):
    def __init__(self, models: List, detach=False):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.detach = detach

    def forward(self, x):
        outputs = []
        for model in self.models:
            _out = model(x)
            outputs.append(_out)
        outputs = torch.stack(outputs, dim=1).reshape(-1, len(self.models))
        return outputs.mean(-1), outputs.std(-1)


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(self,
            state_size=9, action_size=3, n_hidden=16, tau=0.05) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.policies = []
        self.targets = []

        self.criterion = nn.SmoothL1Loss()

        # for idx in range(ensemble_size):
        #     random_seed = random.randint(1000, 100000)
        #     torch.manual_seed(random_seed)
        #     policy_net = DQN(state_size=state_size,
        #                      action_size=action_size,
        #                      n_hidden=n_hidden)
        #     torch.manual_seed(89)
        #     target_net = DQN(state_size=state_size,
        #                      action_size=action_size,
        #                      n_hidden=n_hidden)
        #     target_net.load_state_dict(policy_net.state_dict())
        #     self.policies.append(policy_net)
        #     self.targets.append(target_net)
        random_seed = random.randint(1000, 100000)
        torch.manual_seed(random_seed)
        self.policy = DQN(state_size=state_size,
                          action_size=action_size,
                          n_hidden=n_hidden)
        torch.manual_seed(89)
        self.target = DQN(state_size=state_size,
                          action_size=action_size,
                          n_hidden=n_hidden)
        self.target.load_state_dict(self.policy.state_dict())

    def set_model_id(self, model_id):
        self.policy = self.policies[model_id]
        self.target = self.targets[model_id]

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """

        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        output = self.policy(x)
        return output

    def training_step(self, batch, nb_batch):
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.policy.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.hparams.tau + \
                                         target_net_state_dict[
                                             key] * (1 - self.hparams.tau)
        self.target.load_state_dict(target_net_state_dict)

        state_action_batch, reward = batch
        state_action_values = self.policy(state_action_batch)

        expected_state_action_values = reward
        loss = self.criterion(state_action_values, expected_state_action_values)

        self.log_dict(
            {
                "train_loss": loss,
            }
        )
        self.log("steps", self.global_step, logger=False, prog_bar=True)
        return loss

    def get_ensemble(self):
        return EnsembledModel(self.policies)
