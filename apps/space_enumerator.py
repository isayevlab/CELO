import os

import streamlit as st
import yaml
from lightning.app.components import ServeStreamlit


# from modules.space_generator import SpaceGenerator


class SpaceEnumerator(ServeStreamlit):
    def render(self):
        experiment_name = st.text_input(label="Experiment ID", placeholder="polymer_space_1")
        space_size = st.number_input(label="Max space size", value=10000)
        space_type = st.radio("Select chemical space description type:",
                              ["Custom variables description",
                               "Molecule list"])

        save_folder = None
        if experiment_name is not None:
            save_folder = f"./experiments/{experiment_name}"
            os.makedirs(save_folder, exist_ok=True)

        if space_type == "Custom variables description":
            input_text = st.text_area(label="Space description in YML style")
            input_file = st.file_uploader(label="Upload space descriptor", )
            space_dict = None
            if input_text is not None:
                space_dict = yaml.safe_load(input_text)
            elif input_file is not None:
                space_dict = yaml.safe_load(input_file)

            if space_dict is not None:
                from modules.space_generator import SpaceGenerator
                space = SpaceGenerator(features_dict=space_dict, max_space=space_size)
                space_df = space.space
                if experiment_name is not None:
                    space_df.to_csv(f"{save_folder}/space.csv", index=False)
                st.download_button("Download CSV of constructed space",
                                   space_df.to_csv(),
                                   f"{experiment_name}_space.csv",
                                   type="primary")
        else:
            st.write("lol")
