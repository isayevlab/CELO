import pickle
import pickle as pkl

from abc import abstractmethod, ABC


class Featurizer(ABC):
    """Abstract class for featurizer."""

    @abstractmethod
    def calculate_features(self):
        pass

    def get_features(self):
        return self.features

    def save(self, path):
        """
        Saves features to a file.

        Parameters
        ----------
        path : str
            Path to the save file.

        """
        pkl.dump(self, open(path, "wb"))

    @classmethod
    def load(cls, load_path):
        """
        Loads features and returns it as a class object.

        Parameters
        ----------
        load_path : str
            Path to model's save file.


        Returns
        -------
        :class:'~experiment_flow.featurizer.Featurizer'
            Object of this :class:'~experiment_flow.featurizer.Featurizer'
        """
        return pickle.load(open(load_path, "rb"))
