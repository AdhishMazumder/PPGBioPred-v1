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
# Specify the path to your image file
PPARG_Role = './Utils/Pictures/PPARG-Role.png'

# Open the image in binary mode ('rb')
PPARG = Image.open(PPARG_Role)

# ---Home Page--- #
def about():
    st.header("About PPGBioPred")
    # Picture Upload
    with st.container():
        left_col, right_col = st.columns(2)
        left_col.image(PPARG, output_format="PNG", channels="RGB", width=600)

        left_col2, right_col3 = st.columns(2)
        left_col2.caption('Ahmadian, M. et al. Nature Medicine. 2013.')

        left_col2.caption('**Figure. The figure highlights the role of PPARγ across spectrum of disease targets.** '
               'PPARγ facilitates adipogenesis but at the same time its activation has an adverse effect on the '
               'bone health. Targeting it to enhance bone health could be '
               'a proposed therapeutic rationale for drug development.')

        right_col.markdown(
            f"""
                <h6 style='text-align: justify; color: black; font-weight: normal;'>
                This web server provides a machine learning model for predicting the bioactivity of drugs against 
                PPARγ. PPARγ is an important target for drug discovery, and this model can 
                be used to identify new drug candidates for a variety of diseases, including osteoporosis, cancer, and 
                Alzheimer's disease. This web server is a valuable resource for drug discovery researchers and clinicians. 
                It can be used to identify new drug candidates for a variety of diseases, and it can also be used to 
                predict the bioactivity of existing drugs. 
                </h6>
                <h6 style='text-align: justify; color: black;font-weight: normal;'>
                The server is built using classification and regression models, trained on the most recent dataset 
                obtained from the 
                ChEMBL database, and it can predict the bioactivity of new drugs based on the trained model. 
                To use the model, simply submit the structure of a chemical structure of the drug of your choice
                in the SMILES notation and the model will predict its bioactivity against PPARγ. Upon finishing,
                the server will display the bioactivity values obtained by it. 
                </h6>
                <h6 style='text-align: justify; color: black;font-weight: normal;'>
                We hope that this web server will be a valuable tool for the 
                scientific community, and we are committed to keeping it up-to-date with the latest advances in machine 
                learning and drug discovery.
                </h6>        
                """, unsafe_allow_html=True
    )
