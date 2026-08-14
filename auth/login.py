import streamlit as st

from api_client.auth_client import login_user
from auth.session import login


def login_page():

    st.title("BridgeBot Login")

    username = st.text_input("Username")

    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if not username or not password:

            st.error("Please enter username and password.")

            return

        try:

            response = login_user(username=username, password=password)

            if response:

                user = response["user"]

                # Existing Streamlit session handling
                login(user)

                st.session_state.user = user["username"]
                st.session_state.role = user["role"]
                st.session_state.department = user["department"]
                st.session_state.team = user["team"]
                st.session_state.user_id = user["id"]

                st.success("Login Successful")

                st.rerun()

            else:

                st.error("Invalid Username or Password")

        except Exception as ex:

            st.error(
                f"Login service unavailable: {ex}"
            )