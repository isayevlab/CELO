import pandas as pd
import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from modules.preprocess_data import preprocess_data


class TableDatamodule(LightningDataModule):
    def __init__(self, space_path, labeled_path, target_names, batch_size=32, p=0.1):
        super().__init__()
        if isinstance(target_names, str):
            target_names = target_names.split(", ")
        space = pd.read_csv(space_path, index_col=0)
        labeled_data = pd.read_csv(labeled_path)

        self.features = list(space.columns)

        y = labeled_data.loc[:, target_names]
        y = y.mean(axis=1)

        x = labeled_data.loc[:, self.features]
        x = preprocess_data(x)

        data = train_test_split(x, y, test_size=p) if p is not None else (x, y, x, y)
        self.x_train, self.x_test, self.y_train, self.y_test = data
        self.space = preprocess_data(space)
        self.batch_size = batch_size

    def get_dataset(self, x_df, y_df):
        return TensorDataset(torch.from_numpy(x_df.values),
                             torch.from_numpy(y_df.values))

    def train_dataloader(self):
        data = self.get_dataset(self.x_train, self.y_train)
        return DataLoader(data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        data = self.get_dataset(self.x_test, self.y_test)
        return DataLoader(data, batch_size=self.batch_size)

    def test_dataloader(self):
        data = self.get_dataset(self.x_test, self.y_test)
        return DataLoader(data, batch_size=self.batch_size)

    def predict_dataloader(self):
        space = TensorDataset(torch.from_numpy(self.space.values))
        return DataLoader(space, batch_size=self.batch_size)
