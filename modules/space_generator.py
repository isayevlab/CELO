import numpy as np
import yaml
import pandas as pd

from sklearn.model_selection import ParameterGrid


class SpaceGenerator:
    def __init__(self, features_dict, max_space=10 ** 7, save_space=True):
        self.features = self.build_features(features_dict)
        self.space_size = np.prod([len(features_dict[i]) for i in features_dict])
        self.max_space = max_space
        self.save_space = save_space
        if self.save_space:
            self.construct_space()

    def build_features(self, feature_dict):
        res = {}
        for key, value in feature_dict.items():
            if isinstance(value, list):
                res[key] = value
            elif isinstance(value, dict):
                if "type" in value:
                    if value["type"] == "range":
                        res[key] = np.linspace(*value["params"])
                else:
                    raise NotImplemented("The dictionary should contain 'type' key")
        return res

    def sample(self, n=1):
        if self.save_space:
            return np.random.choice(self.space, n)
        else:
            results = []
            for _ in range(n):
                sample = {}
                for i, k in self.features.items():
                    sample[i] = np.random.choice(k)
                results.append(sample)
            return results

    def construct_space(self):
        if self.max_space >= self.space_size:
            self.space = pd.DataFrame(list(ParameterGrid(self.features)))
        else:
            save_space = self.save_space
            self.save_space = False
            self.space = pd.DataFrame(self.sample(self.max_space))
            self.save_space = save_space

    @classmethod
    def read_yaml(cls, path, **kwargs):
        with open(path) as f:
            my_dict = yaml.safe_load(f)
        return cls(my_dict, **kwargs)


if __name__ == "__main__":
    features_dict = {"a": [1, 2, 3], "b": np.linspace(0, 1, 101)}
    space_generator = SpaceGenerator(features_dict, save_space=True, max_space=10)
    print(space_generator.sample(5))

    space_generator = SpaceGenerator.read_yaml("../../data/example.yaml")

    print(space_generator.sample(5))
