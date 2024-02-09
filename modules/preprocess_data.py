from typing import List, Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(data: pd.DataFrame, scaling_columns: List[Any] = None):
    if scaling_columns is not None:
        minmax_transformer = Pipeline(steps=[
            ('minmax', MinMaxScaler())])

        preprocessor = ColumnTransformer(
            remainder='passthrough',
            transformers=[
                ('mm', minmax_transformer, scaling_columns)
            ])
        preprocessor.set_output(transform="pandas")
        data = preprocessor.fit_transform(data)
    return data
