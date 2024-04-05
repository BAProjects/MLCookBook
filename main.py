# Importing modules
import streamlit as st
from Database.database import update_model_database
from homepage import homepage_content
from HTML_functions import set_background_SB
from maincontent import main_content

# Page configuration
st.set_page_config(page_title="ML Cook Book", page_icon=":trident:", layout="wide", initial_sidebar_state="expanded")

# updating database
update_model_database('models.db')

# Set background image for sidebar
set_background_SB('images\splash.jpg')

# Sidebar content
with st.sidebar:
    st.title('Navigation')
    st.divider()
    selected_page = st.radio("Go to", ("Home", "Models"))
    st.divider()


# Main content
if selected_page == "Models":
    # display Model information
    main_content()
else:
    # Display homepage content
    homepage_content()
