# ---Import Libraries--- #

# Streamlit Libraries
import streamlit as st

# Image Libraries
from PIL import Image

# Warning Library
import warnings

# ---Ignoring any warnings--- #
warnings.simplefilter(action="ignore", category=FutureWarning)

# ---Image Upload Section--- #
PI = Image.open('Dr_I_Manjubala.png')
AD = Image.open('Adhish_Mazumder.png')
NA = Image.open('Nayan.png')

# ---Contact Page--- #

def contact():
    st.markdown("<h3 style='text-align: center; color: black; font-weight:bold;'>Core Members of PPGBioPred</h3>",
                unsafe_allow_html=True)
    st.markdown(
        f"""
            <h6 style='text-align: center; color: black; font-weight:normal;'>
            Please feel free to contact us with any issues, comments, or questions. Your suggestions on improving our 
            server will always be appreciated.
            </h6>
            """, unsafe_allow_html=True
    )

    left_col, right_col = st.columns(2)

    left_col.image(PI, output_format="PNG", channels="RGB", width=280)
    left_col.markdown("<h4 style='text-align: justify; color: black; font-weight:bold;'>Dr. I. Manjubala</h4>",
                      unsafe_allow_html=True)
    left_col.markdown(
        "<h5 style='text-align: justify; color: maroon; font-weight:bold;'>Principle Investigator</h5>",
        unsafe_allow_html=True)
    left_col.markdown(
        f"""
            <h6 style='text-align: justify; color: black; font-weight:normal;'>
            Biomaterials and Bone Research Laboratory
            </h6>
            <h6 style='text-align: justify; color: black; font-weight:normal;'>
            School of Bio Sciences and Technology
            </h6>
            """, unsafe_allow_html=True
    )
    left_col.markdown(
        """
        [![Email](<https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Circle-icons-mail.svg/50px-Circle-icons-mail.svg.png>)](<i.manjubala@vit.ac.in>)        
        """
    )

    right_col.image(AD, output_format="PNG", channels="RGB", width=290)
    right_col.markdown("<h4 style='text-align: justify; color: black; font-weight:bold;'>Adhish Mazumder</h4>",
                       unsafe_allow_html=True)
    right_col.markdown("<h5 style='text-align: justify; color: maroon; font-weight:bold;'>Researcher</h5>",
                       unsafe_allow_html=True)
    right_col.markdown(
        f"""
            <h6 style='text-align: justify; color: black; font-weight:normal;'>
            Biomaterials and Bone Research Laboratory
            </h6>
            <h6 style='text-align: justify; color: black; font-weight:normal;'>
            School of Bio Sciences and Technology
            </h6>
            """, unsafe_allow_html=True
    )

    right_col.markdown("""
            [![Email](<https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Circle-icons-mail.svg/50px-Circle-icons-mail.svg.png>)](<adhish.mazumder@vit.ac.in>)
            [![LinkedIn](<https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/LinkedIn_icon_circle.svg/50px-LinkedIn_icon_circle.svg.png>)](<https://www.linkedin.com/in/adhishmazumder/>)
            [![GitHub](<https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/50px-Octicons-mark-github.svg.png>)](<https://github.com/AdhishMazumder>)
            """)

    st.markdown('---')

    st.markdown("<h3 style='text-align: center; color: black; font-weight:bold;'>Associated Project Members</h3>",
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    col1.image(NA, output_format="PNG", channels="RGB", width=290)
    col1.markdown("<h4 style='text-align: justify; color: black; font-weight: bold;'>Nayan Chaudhari</h4>",
                  unsafe_allow_html=True)
    col1.markdown("<h5 style='text-align: justify; color: maroon; font-weight: bold'>B.Tech Student</h5>",
                  unsafe_allow_html=True)
    col1.markdown(
        f"""
            <h6 style='text-align: justify; color: black; font-weight:normal;'>
            School of Bio Sciences and Technology
            </h6>
            <h6 style='text-align: justify; color: black; font-weight:bold;'>
            2023
            </h6>
            """, unsafe_allow_html=True
    )

    st.markdown('---')
