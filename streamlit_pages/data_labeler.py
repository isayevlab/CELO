import os
import pandas as pd
import streamlit as st


def data_labeler():
    st.title("Data Labeler")

    experiment_name = st.text_input("Experiment ID", placeholder="Enter the experiment ID, e.g., polymer_space_1")
    if experiment_name:
        space_path = os.path.join("experiments", experiment_name, "space.csv")
        label_path = os.path.join("experiments", experiment_name, "labeled_samples.csv")

        if os.path.exists(space_path):
            space_df = pd.read_csv(space_path, index_col=0)

            # Load or initialize labeled data
            if os.path.exists(label_path):
                labeled_df = pd.read_csv(label_path, index_col=0)
            else:
                labeled_df = pd.DataFrame()

            # Section 1: Add Indexes
            st.subheader("Add Indexes")

            # Multiselect option for IDs
            selected_ids_multi = st.multiselect("Select Data Point IDs", options=list(space_df.index))

            # Text input option for IDs (space-separated)
            selected_ids_text = st.text_input("Or enter Data Point IDs separated by space")

            if selected_ids_text:
                selected_ids_text_list = selected_ids_text.split()
                if pd.api.types.is_numeric_dtype(space_df.index):
                    selected_ids_text_list = [int(i) for i in selected_ids_text_list]
                selected_ids = selected_ids_multi + [idx for idx in selected_ids_text_list if idx not in selected_ids_multi]
            else:
                selected_ids = selected_ids_multi

            if not pd.api.types.is_numeric_dtype(space_df.index):
                selected_ids = [str(id) for id in selected_ids]

            # Find only available IDs to avoid KeyError
            available_ids = space_df.index.intersection(selected_ids)
            missing_ids = set(selected_ids) - set(available_ids)

            # Exclude already labeled IDs
            available_ids = available_ids.difference(labeled_df.index)

            # Button to add new indexes
            if st.button("Add Selected Indexes"):
                if not available_ids.empty:
                    new_data = space_df.loc[available_ids]
                    labeled_df = pd.concat([labeled_df, new_data], axis=0)
                    labeled_df.to_csv(label_path, index=True)
                    st.success("Selected indexes added successfully. Refresh the page to view the updated data.")
                    if missing_ids:
                        st.warning(f"The following IDs were not found in the data: {', '.join(map(str, missing_ids))}")
                else:
                    st.error("Please select valid IDs to add.")

            # Section 2: Remove Indexes
            st.subheader("Remove Indexes")

            # Select indexes to remove
            removable_ids = st.multiselect("Select Data Point IDs to Remove", options=list(labeled_df.index))
            if st.button("Remove Selected Indexes") and removable_ids:
                labeled_df = labeled_df.drop(index=removable_ids)
                labeled_df.to_csv(label_path, index=True)
                st.success("Selected indexes removed successfully. Refresh the page to view the updated data.")

            # Section 3: Add Columns
            st.subheader("Add Columns")

            # Add new label columns
            new_label_columns = st.text_input("Enter new label columns, separated by commas (e.g., Stability, Solubility)")
            if st.button("Add New Columns"):
                if new_label_columns:
                    new_label_columns = [col.strip() for col in new_label_columns.split(',')]
                    for col in new_label_columns:
                        labeled_df[col] = pd.NA  # Add new columns with NA values
                    labeled_df = pd.concat([labeled_df[new_label_columns], labeled_df.drop(columns=new_label_columns)], axis=1)
                    labeled_df.to_csv(label_path, index=True)
                    st.success("New columns added successfully. Refresh the page to view the updated data.")
                else:
                    st.error("Please enter at least one column name.")

            # Section 4: Remove Columns
            st.subheader("Remove Columns")
            remove_column = st.selectbox("Select column to remove", options=labeled_df.columns, index=0) if not labeled_df.empty else None
            if st.button("Remove Selected Column"):
                if remove_column:
                    labeled_df = labeled_df.drop(columns=[remove_column])
                    labeled_df.to_csv(label_path, index=True)
                    st.success(f"Column '{remove_column}' removed successfully. Refresh the page to view the updated data.")

            # Editable DataFrame
            st.subheader("Labeled Data (Editable)")
            editable_df = st.data_editor(labeled_df)

            # Save changes made in the editable dataframe
            if st.button("Save Changes"):
                editable_df.to_csv(label_path, index=True)
                st.success("Changes saved successfully. Refresh the page to view the updated data.")

        else:
            st.error(f"Experiment ID '{experiment_name}' not found. Please verify the ID and try again.")


if __name__ == "__main__":
    data_labeler()
