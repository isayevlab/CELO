import streamlit as st
import streamlit_authenticator as stauth
# from streamlit_authenticator.utilities.hasher import Hasher

from configs.credentials import credentials

def main():
    # No hashing for now, can add it later for security
    # # Hash the passwords
    # for username, user_data in credentials["usernames"].items():
    #     user_data["password"] = Hasher([user_data["password"]]).generate()[0]

    authenticator = stauth.Authenticate(
        credentials, "celo_cookies", "celo_signature", cookie_expiry_days=30
    )

    name, authentication_status, username = authenticator.login("main")

    if authentication_status:
        st.write(f"Welcome *{name}*")
        authenticator.logout("Logout", "sidebar")

        # Add custom CSS to make radio buttons bigger and bold
        st.sidebar.markdown(
            """
            <style>
            .sidebar .radio-text {
                font-size: 20px;
                font-weight: bold;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Define the app pages
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Go to",
            ["Space Enumerator", "Sample Selector", "Data Labeler", "ML Model Selector", "Explorator Explotator"],
            format_func=lambda x: "🔍 " + x if x == "Space Enumerator" else (
                "📊 " + x if x == "Sample Selector" else (
                    "🏷️ " + x if x == "Data Labeler" else (
                        "🤖 " + x if x == "ML Model Selector" else (
                            "🧪 " + x if x == "Explorator Explotator" else x
                        )
                    )
                )
            ),
            key="main_menu"
        )

        if page == "Space Enumerator":
            from streamlit_pages import space_enumerator
            space_enumerator.space_enumerator()
        elif page == "Sample Selector":
            from streamlit_pages import sample_selector
            sample_selector.sample_selector()
        elif page == "Data Labeler":
            from streamlit_pages import data_labeler
            data_labeler.data_labeler()
        elif page == "ML Model Selector":
            from streamlit_pages import ml_model_selector
            ml_model_selector.ml_model_selector()
        elif page == "Explorator Explotator":
            from streamlit_pages import explorator_explotator
            explorator_explotator.explorator_explotator()

    elif authentication_status == False:
        st.error("Username/password is incorrect")

    elif authentication_status == None:
        st.warning("Please enter your username and password")

if __name__ == "__main__":
    main()
