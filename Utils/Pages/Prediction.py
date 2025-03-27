# ---Import Libraries--- #

# Streamlit Libraries
import streamlit as st

# Core Libraries
import base64
import pickle
import subprocess

# Powering Libraries
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from stmol import showmol
import matplotlib.pyplot as plt
import py3Dmol

# Image Libraries
from PIL import Image

# Warning Library
import warnings

# ---Ignoring any warnings--- #
warnings.simplefilter(action="ignore", category=FutureWarning)

# ---Bioactivity Predictions Page--- #

def pred():
    col1, col2 = st.columns(2)
    col1.markdown("### **Submit the Compound**")
    st.markdown("---")

    # Defining the molecules from smiles data and rendering its visualization

    def makeblock(smi):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mblock = Chem.MolToMolBlock(mol)
        return mblock

    def render_mol(xyz):
        xyzview = py3Dmol.view(height=250, width=650)
        xyzview.addModel(xyz, 'mol')
        xyzview.setStyle({'stick': ()})
        xyzview.setBackgroundColor('white')
        xyzview.zoomTo()
        showmol(xyzview, height=250, width=650)

    # Submission Process
    with col1:
        compound_smiles = col1.text_input('Enter your SMILES data',
                                          'CC(=O)OC1=CC=CC=C1C(=O)O')
        blk = makeblock(compound_smiles)
        compound_name = col1.text_input('Enter the Compound Name/ID', 'Aspirin')

    with col2:
        render_mol(blk)

    col1, col2 = st.columns(2)
    compound_pathway = col1.selectbox(
        'Select the bioactivity to be predicted',
        ('IC50', 'EC50')
        , index=0)

    # Target Specification for Pathways
    if compound_pathway == "IC50":
        wnt_targets = col2.selectbox(
            'Select the target',
            ('PPARγ',), index=0)
        st.markdown("---")

        # Button functionality
        if st.button("Submit for Prediction", key='Submit'):
            submission_data = pd.DataFrame(
                {"SMILES": [compound_smiles], "Compound Name/ID": [compound_name]})
            submission_data.to_csv('molecule.smi', sep='\t', header=False,
                                   index=False)
            if wnt_targets == "PPARγ":
                st.markdown('## **Prediction Results**')

                # Molecular descriptor calculator
                def desc_calc():
                    # Specify the full command as a single string
                    bashCommand_reg = (
                        "java -Xms2G -Xmx2G -Djava.awt.headless=true "
                        "-jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar "
                        "-removesalt -standardizenitro -fingerprints "
                        "-descriptortypes ./PaDEL-Descriptor/SubstructureFingerprinter.xml "
                        "-dir ./ -file descriptors_output_pic50_RF.csv"
                    )

                    try:
                        process_reg = subprocess.Popen(bashCommand_reg, shell=True, stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE)
                        output_reg, error_reg = process_reg.communicate()

                        if process_reg.returncode == 0:
                            print("Descriptor calculation completed successfully.")
                        else:
                            print(f"Error encountered:\n{error_reg.decode('utf-8')}")
                    except Exception as e:
                        print(f"An error occurred: {e}")
                    # os.remove('molecule.smi')

                # File download
                def filedownload(df):
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(
                        csv.encode()).decode()  # strings <-> bytes conversions
                    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download </a>'
                    return href

                def build_model_reg(input_data, compound_name):
                    # Load download icon image with error handling
                    download_icon_path = './Utils/Pictures/Download-Icon.png'
                    if not os.path.exists(download_icon_path):
                        st.error(f"Download icon not found at: {download_icon_path}")
                        return
                    downloadbutton = Image.open(download_icon_path)
                
                    # Load the regression model with error handling
                    rf_subc_ic50 = './Utils/Pages/Models/Regression/IC50/rf_reg_sub_model.pkl'
                    if not os.path.exists(rf_subc_ic50):
                        st.error(f"Regression model file not found at: {rf_subc_ic50}")
                        return
                    try:
                        with open(rf_subc_ic50, 'rb') as model_file:
                            load_model = pickle.load(model_file)
                    except Exception as e:
                        st.error(f"Error loading the regression model: {e}")
                        return
                
                    # Make predictions with error handling
                    try:
                        prediction_reg = load_model.predict(input_data)
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        return
                
                    st.markdown('###### **$IC_{50}$**')
                    prediction_output_reg = pd.Series(prediction_reg, name='pIC50')
                    molecule_name = pd.Series(compound_name, name='Compound Name/ID')
                
                    # Convert pIC50 to IC50 (in M) and then to nM
                    calc_IC50 = 10 ** (-prediction_output_reg) * 1e9  # 10^(-pIC50) in M, multiplied by 1e9 to convert to nM
                    prediction_IC50 = pd.Series(calc_IC50, name='IC50 (nM)')
                
                    df = pd.concat([molecule_name, prediction_output_reg, prediction_IC50], axis=1)
                
                    st.markdown(
                        "The **$IC_{50}$** of " + str(compound_name) + " is " +
                        "<span style='color:green'><b>" + str(np.round(calc_IC50.iloc[0], 2)) +
                        " nM</b></span>", unsafe_allow_html=True)
                    st.write(df)
                
                    # Create columns to display download icon and file download link
                    columns = st.columns(15)
                    with columns[0]:
                        st.image(downloadbutton, output_format="PNG", channels="RGB", width=25)
                    with columns[1]:
                        st.markdown(filedownload(df), unsafe_allow_html=True)
                
                    return df
                
                # Example usage within the app:
                with st.spinner("PPGBioPred is calculating Bioactivity..."):
                    # Assuming desc_calc() is a function that calculates descriptors
                    desc_calc()
                    
                    # Check if the descriptors CSV file exists before attempting to read it
                    descriptors_file = 'descriptors_output_pic50_RF.csv'
                    if not os.path.exists(descriptors_file):
                        st.error(f"Descriptors file not found: {descriptors_file}. Please verify the file path.")
                        st.stop()
                    try:
                        desc_reg = pd.read_csv(descriptors_file)
                    except Exception as e:
                        st.error(f"Error reading the descriptors file: {e}")
                        st.stop()
                
                    # Check and load the descriptor list used in the regression model
                    descriptor_list_file = './Utils/Pages/Models/Regression/IC50/df_Substructure_final.csv'
                    if not os.path.exists(descriptor_list_file):
                        st.error(f"Descriptor list file not found: {descriptor_list_file}")
                        st.stop()
                    try:
                        Xlist_reg = list(pd.read_csv(descriptor_list_file).columns)
                    except Exception as e:
                        st.error(f"Error reading the descriptor list file: {e}")
                        st.stop()
                
                    # Subset the descriptors based on the expected list
                    try:
                        desc_subset_reg = desc_reg[Xlist_reg]
                    except KeyError as e:
                        st.error(f"Error subsetting descriptors: {e}")
                        st.stop()
                
                    # Ensure compound_name is defined or obtained from your app context
                    compound_name = "YourCompoundName"  # Replace with actual compound name or variable
                    build_model_reg(desc_subset_reg, compound_name)

