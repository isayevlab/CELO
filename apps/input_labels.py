import os

import pandas as pd
import streamlit as st
from lightning.app.components import ServeStreamlit


class InputLabel(ServeStreamlit):
    def render(self):
        experiment_name = st.text_input(label="Experiment ID", placeholder="polymer_space_1")
        if (experiment_name is not None) and (len(experiment_name) != 0):
            space_path = f"experiments/{experiment_name}/space.csv"
            if (os.path.exists(f"experiments/{experiment_name}")
                    and os.path.exists(space_path)):
                space = pd.read_csv(space_path, index_col=0)
                features = list(space.columns)
                label_path = f"experiments/{experiment_name}/labeled_samples.csv"
                if os.path.exists(label_path):
                    labeled_data = pd.read_csv(label_path)
                else:
                    labeled_data = pd.DataFrame(columns=features)

                edited_df = st.data_editor(labeled_data,
                                           num_rows="dynamic")
                if st.button("Rewrite Labeled Data", type="primary"):
                    edited_df = edited_df.fillna(0)
                    print(edited_df)
                    edited_df.to_csv(label_path, index=False)
            else:
                st.write(f"There is no space for - {experiment_name}")
