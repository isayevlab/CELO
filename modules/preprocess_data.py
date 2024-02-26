import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(data: pd.DataFrame):
    min_values = data.min(axis=0)
    max_values = data.min(axis=0)
    scaling_columns = data.columns[(min_values < 0.) | (max_values > 1.)]
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
