import os
import streamlit as st
import yaml
import pandas as pd
from streamlit_ace import st_ace

def get_index(df):
    result = []
    for i, q in df.iterrows():
        q = q.to_dict()

        s = ""
        for x, y in q.items():
            x = f"{x}"
            if not isinstance(y, float):
                y = f"{y}"
            else:
                y = f"{y:.2f}"
            s += f"{x}_{y}_"
        result.append(s)
    return result

def space_enumerator():
    experiment_name = st.text_input(label="Experiment ID", placeholder="polymer_space_1")
    space_size = st.number_input(label="Max space size", value=10000)

    st.header("Space description in YML style")
    input_text = st_ace("", language="yaml", )
    input_file = st.file_uploader(label="Upload space descriptor")
    input_features = st.file_uploader(label="Upload molecule features",
                                      accept_multiple_files=True)
    space_dict = None
    if input_text is not None and len(input_text) > 0:
        space_dict = yaml.safe_load(input_text)
    elif input_file is not None:
        space_dict = yaml.safe_load(input_file)
    if space_dict is not None:
        from modules.space_generator import SpaceGenerator
        name_to_file = {}
        if input_features is not None:
            for file in input_features:
                name_to_file[file.name] = file
        space = SpaceGenerator(features_dict=space_dict, max_space=space_size,
                               save_space=True, name_to_files=name_to_file)
        space_df = space.space
        if len(space_df.columns) < 50:
            space_df["index"] = get_index(space_df)
            space_df = space_df.set_index("index")
        if experiment_name is not None and len(experiment_name) > 0:
            save_folder = f"./experiments/{experiment_name}"
            os.makedirs(save_folder, exist_ok=True)
            space_df.to_csv(f"{save_folder}/space.csv")
        st.dataframe(space_df)
        st.download_button("Download CSV of constructed space",
                           space_df.to_csv(),
                           f"{experiment_name}_space.csv",
                           type="primary")

if __name__ == "__main__":
    space_enumerator()
