import itertools
import random

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid


def weights_decomposition(K, total_sum=1., step=0.1, weights=None, shuffle=False):
    if weights is None:
        weights = []

    if K == 1:
        if total_sum > 0:
            weights.append(total_sum)
            yield weights

    if total_sum < 0:
        return

    sum = 0

    possible_steps = []

    while sum < total_sum:
        sum += step
        if total_sum - sum >= step:
            possible_steps.append(sum)

    if shuffle:
        random.shuffle(possible_steps)

    for sum in possible_steps:
        yield from weights_decomposition(K - 1, total_sum - sum, step=step,
                                         weights=[sum] + weights, shuffle=shuffle)


class ComponentsMixture:
    def __init__(self, components_names, max_num_component=None, precision=0.1, shuffle=True):
        self.names = components_names
        self.max_num_component = max_num_component if max_num_component is not None else \
            len(self.names)
        self.combs, self.weights = self.space_enumeration(precision, shuffle)
        self.shuffle = shuffle

    def space_enumeration(self, precision, shuffle):
        combs = list(itertools.combinations(self.names, self.max_num_component))
        weights = list(weights_decomposition(self.max_num_component, total_sum=1.,
                                        step=precision, shuffle=shuffle))

        if shuffle:
            random.shuffle(combs)
        return combs, weights

    def sample(self, n):

        if n <= len(self.combs):
            combs = np.random.choice(self.combs, n, replace=False)
            weights = np.random.choice(self.weights, n)
        else:
            combs = np.random.choice(self.combs, n)
            weights = np.random.choice(self.weights, n)
        return combs, weights


class SpaceGenerator:
    def __init__(self, features_dict, max_space=10 ** 7, save_space=True):
        self.grid_feats, self.mix_feats = self.build_features(features_dict)
        space_size = np.prod([len(i) for i in self.grid_feats.values()])

        self.space_size = np.prod([len(v.weights)*len(v.combs) for v in self.mix_feats.values()])*space_size

        self.max_space = max_space
        self.save_space = save_space
        if self.save_space:
            self.construct_space()

    def build_features(self, feature_dict):
        grid_features = {}
        mixture_features = {}
        for key, value in feature_dict.items():
            if isinstance(value, list):
                grid_features[key] = value
            elif isinstance(value, dict):
                if "type" in value:
                    if value["type"] == "range":
                        grid_features[key] = np.linspace(*value["params"])
                    elif value["type"] == "mixture":
                        mixture_features[key] = ComponentsMixture(value["components"], **value["params"])
                else:
                    raise NotImplemented("The dictionary should contain 'type' key")

        return grid_features, mixture_features

    def sample(self, n=1):
        if self.save_space:
            return np.random.choice(self.space, n)
        else:
            results = []
            for _ in range(n):
                sample = {}
                for i, k in self.grid_feats.items():
                    sample[i] = np.random.choice(k)

                for mixture in self.mix_feats.values():
                    idx = np.random.randint(len(mixture.combs))
                    comb = mixture.combs[idx]
                    idx = np.random.randint(len(mixture.weights))
                    weight = mixture.weights[idx]
                    sample = {**sample, **dict(zip(comb, weight))}
                results.append(sample)
            return results

    def construct_space(self):
        if self.max_space >= self.space_size:
            param_grid = list(ParameterGrid(self.grid_feats))

            names = list(param_grid[0].keys())

            for mixture in self.mix_feats.values():
                names += mixture.names
            space = list()
            for param in param_grid:
                for mixture in self.mix_feats.values():
                    for comb in mixture.combs:
                        for weight in mixture.weights:
                            dct = {**param, **dict(zip(comb, weight))}
                            space.append(dct)
            self.space = pd.DataFrame(space).fillna(0.)


        else:
            save_space = self.save_space
            self.save_space = False
            self.space = pd.DataFrame(self.sample(self.max_space)).fillna(0.)
            self.save_space = save_space

    @classmethod
    def read_yaml(cls, path, **kwargs):
        with open(path) as f:
            my_dict = yaml.safe_load(f)
        return cls(my_dict, **kwargs)


if __name__ == "__main__":
    features_dict = {"a": [1, 2, 3], "b": {"type": "range", "params": [0, 1, 11]},
                     "C": {"type": "mixture", "components": ["A", "B", "C", "D"],
                           "params": {"max_num_component": 3}}}
    space_generator = SpaceGenerator(features_dict, save_space=False, max_space=5000)

    print(space_generator.sample(5))
