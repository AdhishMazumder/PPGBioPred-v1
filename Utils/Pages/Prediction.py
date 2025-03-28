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
import joblib

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

    # Function to generate molecule block for visualization
    def makeblock(smi):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mblock = Chem.MolToMolBlock(mol)
        return mblock

    def render_mol(xyz):
        xyzview = py3Dmol.view(height=250, width=650)
        xyzview.addModel(xyz, 'mol')
        xyzview.setStyle({'stick': {}})
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
        ('IC50', 'EC50', 'Activity %'),
        index=0
    )

    # Target Specification for Pathways
    if compound_pathway == "IC50":
        wnt_targets = col2.selectbox(
            'Select the target',
            ('PPARγ',),
            index=0
        )
        st.markdown("---")

        # Button functionality
        if st.button("Submit for Prediction", key='Submit'):
            submission_data = pd.DataFrame(
                {"SMILES": [compound_smiles], "Compound Name/ID": [compound_name]}
            )
            submission_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            if wnt_targets == "PPARγ":
                st.markdown('## **Prediction Results**')

                # Molecular descriptor calculator
                def desc_calc():
                    # Full command to run PaDEL-Descriptor for descriptor calculation
                    bashCommand = (
                        "java -Xms2G -Xmx2G -Djava.awt.headless=true "
                        "-jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar "
                        "-removesalt -standardizenitro -fingerprints "
                        "-descriptortypes ./PaDEL-Descriptor/SubstructureFingerprintCount.xml "
                        "-dir ./ -file descriptors_output_pic50_RF.csv"
                    )
                    
                    try:
                        process = subprocess.Popen(
                            bashCommand, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE
                        )
                        output, error = process.communicate()
                        if process.returncode == 0:
                            print("Descriptor calculation completed successfully.")
                        else:
                            print(f"Error encountered:\n{error.decode('utf-8')}")
                    except Exception as e:
                        print(f"An error occurred: {e}")
                    # Optionally, remove the molecule file
                    # os.remove('molecule.smi')

                # File download helper
                def filedownload(df):
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()  # string <-> bytes conversion
                    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download </a>'
                    return href

                # Build the regression model predictions
                def build_model(input_data, compound_name):
                    download = './Utils/Pictures/Download-Icon.png'
                    downloadbutton = Image.open(download)

                    # Load the saved regression model
                    rf_sub_ic50 = './Utils/Pages/Models/Regression/IC50/rf_reg_sub_model.pkl'
                    load_model = joblib.load(open(rf_sub_ic50, 'rb'))

                    # Apply model to make predictions
                    prediction = load_model.predict(input_data)

                    st.markdown('###### **$IC_{50}$**')
                    prediction_output = pd.Series(prediction, name='pIC50')
                    molecule_name_series = pd.Series(compound_name, name='Compound Name/ID')

                    # Convert pIC50 to IC50 (in M) and then to nM
                    calc_IC50 = 10 ** (-prediction_output) * 1e9  # 10^(-pIC50) in M -> nM conversion
                    prediction_IC50 = pd.Series(calc_IC50, name='IC50 (nM)')

                    df = pd.concat([molecule_name_series, prediction_output, prediction_IC50], axis=1)

                    st.markdown(
                        "The **$IC_{50}$** of " + str(compound_name) + " is " +
                        "<span style='color:green'><b>" + str(np.round(calc_IC50.iloc[0], 2)) +
                        " nM</b></span>", unsafe_allow_html=True
                    )
                    st.write(df)

                    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 = st.columns(15)
                    with c1:
                        st.image(downloadbutton, output_format="PNG", channels="RGB", width=25)
                    with c2:
                        st.markdown(filedownload(df), unsafe_allow_html=True)
                    return df

                with st.spinner("PPGBioPred is calculating Bioactivity..."):
                    # Calculate descriptors using PaDEL-Descriptor
                    desc_calc()

                    # Read in calculated descriptors
                    desc = pd.read_csv('descriptors_output_pic50_RF.csv')
                    
                    # Read descriptor list used in previously built model
                    Xlist = list(pd.read_csv('./Utils/Pages/Models/Regression/IC50/df_Substructure_final.csv').columns)
                    
                    # Instead of dropping columns, select only the columns that are common between the descriptor file and Xlist
                    common_cols = [col for col in Xlist if col in desc.columns]
                    if not common_cols:
                        st.error("None of the expected descriptor columns were found in the descriptor file.")
                        st.stop()
                    desc_subset = desc[common_cols]

                    # Apply the trained regression model to make predictions
                    build_model(desc_subset, compound_name)

                # Molecular descriptor calculator
                def desc_calc_clf():
                    # Full command to run PaDEL-Descriptor for descriptor calculation
                    bashCommand_clf = (
                        "java -Xms2G -Xmx2G -Djava.awt.headless=true "
                        "-jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar "
                        "-removesalt -standardizenitro -fingerprints "
                        "-descriptortypes ./PaDEL-Descriptor/MACCSFingerprintCount.xml "
                        "-dir ./ -file descriptors_output_pic50_CF.csv"
                    )
                    
                    try:
                        process_clf = subprocess.Popen(
                            bashCommand_clf, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE
                        )
                        output_clf, error_clf = process_clf.communicate()
                        if process_clf.returncode == 0:
                            print("Descriptor calculation completed successfully.")
                        else:
                            print(f"Error encountered:\n{error_clf.decode('utf-8')}")
                    except Exception as e:
                        print(f"An error occurred: {e}")
                    # Optionally, remove the molecule file
                    # os.remove('molecule.smi')

                # File download helper
                def filedownload_clf(df_clf):
                    csv_clf = df_clf.to_csv(index=False)
                    b64_clf = base64.b64encode(csv_clf.encode()).decode()  # string <-> bytes conversion
                    href_clf = f'<a href="data:file/csv;base64_clf,{b64_clf}" download="prediction_clf.csv">Download </a>'
                    return href_clf

                # Build the regression model predictions
                def build_model_clf(input_data, compound_name):
                    download = './Utils/Pictures/Download-Icon.png'
                    downloadbutton = Image.open(download)

                    # Load the saved regression model
                    cf_maccs_ic50 = './Utils/Pages/Models/Classification/IC50/rf_maccs_model.pkl'
                    load_model_clf = joblib.load(open(cf_maccs_ic50, 'rb'))

                    # Apply model to make predictions
                    prediction_clf = load_model_clf.predict(input_data)

                    st.markdown('###### **$IC_{50}$**')
                    prediction_output_clf = pd.Series(prediction_clf, name='Classification')
                    molecule_name_series = pd.Series(compound_name, name='Compound Name/ID')

                    df_clf = pd.concat([molecule_name_series, prediction_output_clf], axis=1)

                    st.markdown(
                        "The compound, " + str(compound_name) + " is " +
                        "<span style='color:green'><b>" + str(prediction_output_clf[0]) + "</b></span>" +
                        " against the target.", unsafe_allow_html=True)
                    st.write(df_clf)

                    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 = st.columns(15)
                    with c4:
                        st.image(downloadbutton, output_format="PNG", channels="RGB", width=25)
                    with c5:
                        st.markdown(filedownload_clf(df_clf), unsafe_allow_html=True)
                    return df

                with st.spinner("PPGBioPred is calculating Bioactivity..."):
                    # Calculate descriptors using PaDEL-Descriptor
                    desc_calc_clf()

                    # Read in calculated descriptors
                    desc_clf = pd.read_csv('descriptors_output_pic50_CF.csv')
                    
                    # Read descriptor list used in previously built model
                    Xlist_clf = list(pd.read_csv('./Utils/Pages/Models/Classification/IC50/df_MACCS_final_classification.csv').columns)
                    
                    # Instead of dropping columns, select only the columns that are common between the descriptor file and Xlist
                    common_cols_clf = [col for col in Xlist_clf if col in desc_clf.columns]
                    if not common_cols:
                        st.error("None of the expected descriptor columns were found in the descriptor file.")
                        st.stop()
                    desc_subset_clf = desc_clf[common_cols_clf]

                    # Apply the trained regression model to make predictions
                    build_model_clf(desc_subset_clf, compound_name)
