# ---Import Libraries--- #
# Streamlit Libraries
import streamlit as st

# Folder Libraries
from Utils.Pages.Coreapp import coreapp

# OS Libraries
import os

# Database Libraries
from deta import Deta
from dotenv import load_dotenv

# Image Libraries
from PIL import Image

# Warning Library
import warnings

# ---Ignoring any warnings--- #
warnings.simplefilter(action="ignore", category=FutureWarning)

# ---Page Configuration--- #
Server = './Utils/Pictures/Server.png'
img = Image.open(Server)
st.set_page_config(page_title="PPGBioPred", page_icon=img, layout="wide")

# ---Hiding the Streamlit Header and Footer--- #
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

coreapp()

# ---Custom Footer--- #
st.markdown('---')

with st.container():
    #co1, co2 = st.columns(2)

    #with co1:
    st.markdown(
        f"""
                <h6 style='text-align: center; color: black;font-weight: normal;'>
                Created using
                </h6>
                """, unsafe_allow_html=True)

    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = st.columns(11)
    with x4:
        st.markdown(
            f"""
                    ![Python](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Python_logo_51.svg/50px-Python_logo_51.svg.png?2021051019534)                                                            
                    """)
    with x5:
        st.markdown(
            f"""                                
                    ![Streamlit](https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Streamlit_logo_primary_colormark_darktext.svg/80px-Streamlit_logo_primary_colormark_darktext.svg.png)                                                           
                    """)
    with x6:
        st.markdown(
            f"""                                
                    ![Pandas](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/100px-Pandas_logo.svg.png)                                                           
                    """)
    with x7:
        st.markdown(
            f"""
                    ![Numpy](https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/100px-NumPy_logo_2020.svg.png)                                                            
                    """)
    with x8:
        st.markdown(
            f"""
                    ![Matplotlib](https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Matplotlib_icon.svg/45px-Matplotlib_icon.svg.png)                             
                    """)

st.markdown("---")
