# ---Import Libraries--- #

# Streamlit Libraries
import streamlit as st
from streamlit_option_menu import option_menu

# Image Libraries
from PIL import Image

# Folder Libraries
from Utils.Pages.Homepage import about
from Utils.Pages.Prediction import pred
#from Utils.Pages.Tutorials import tutorial
from Utils.Pages.Contacts import contact

# Warning Library
import warnings

# ---Ignoring any warnings--- #
warnings.simplefilter(action="ignore", category=FutureWarning)

def coreapp():

    # ---Image Upload Section--- #
    Server_Logo = './Utils/Pictures/ServerLogo.png'
    Logo = Image.open(Server_Logo)

    # ---Head Page Configuration--- #
    left_col, right_col = st.columns(2)

    left_col.image(Logo, output_format="PNG", channels="RGB", width=900)

    # ---Hiding the Streamlit Header and Footer--- #
    hide_st_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # ---Setting the Navigation Bar--- #
    select = option_menu(
        menu_title=None,
        options=["Home", "Prediction", "Contact"],
        icons=["house-door-fill", "cloud-upload-fill", "envelope-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#aaf0d1"},
            "icon": {"color": "#cf1020", "font-size": "20px"},
            "nav-link": {"font-size": "20px", "text-align": "center", "margin": "5px",
                         "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#20b2aa"},
        }
    )

    # ---Selection of Pages--- #
    if select == "Home":
        about()

    if select == "Prediction":
        pred()


    if select == "Contact":
        contact()
