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

                    bashCommand_clf = (
                        "java -Xms2G -Xmx2G -Djava.awt.headless=true "
                        "-jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar "
                        "-removesalt -standardizenitro -fingerprints "
                        "-descriptortypes ./PaDEL-Descriptor/MACCSFingerprinter.xml "
                        "-dir ./ -file descriptors_output_pic50_CF.csv"
                    )

                    try:
                        process_reg = subprocess.Popen(bashCommand_reg, shell=True, stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE)
                        process_clf = subprocess.Popen(bashCommand_clf, shell=True, stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE)
                        output_reg, error_reg = process_reg.communicate()
                        output_clf, error_clf = process_clf.communicate()

                        if process_reg.returncode == 0:
                            print("Descriptor calculation completed successfully.")
                        else:
                            print(f"Error encountered:\n{error_reg.decode('utf-8')}")
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        
                        if process_clf.returncode == 0:
                            print("Descriptor calculation completed successfully.")
                        else:
                            print(f"Error encountered:\n{error_clf.decode('utf-8')}")
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

                def build_model_reg(input_data):
                    download = './Utils/Pictures/Download-Icon.png'
                    downloadbutton = Image.open(download)

                    # Reads in saved regression model
                    rf_subc_ic50 = './Utils/Pages/Models/Regression/IC50/rf_reg_sub_model.pkl'
                    load_model = pickle.load(open(rf_subc_ic50, 'rb'))

                    # Apply model to make predictions
                    prediction_reg = load_model.predict(input_data)

                    st.markdown('###### **$IC_{50}$**')
                    prediction_output_reg = pd.Series(prediction_reg, name='pIC50')
                    molecule_name = pd.Series(compound_name, name='Compound Name/ID')

                    # Convert pIC50 to IC50 (in M) and then to nM
                    calc_IC50 = 10 ** -prediction_output * 10 ** 9
                    prediction_IC50 = pd.Series(calc_IC50, name='IC50 (nM)')

                    df = pd.concat([molecule_name, prediction_output_reg, prediction_IC50], axis=1)

                    st.markdown(
                        "The **$IC_{50}$** of " + str(compound_name) + " is " + "<span style='color:green'><b>" +
                        str(np.round(calc_IC50[0], 2)) + " nM", unsafe_allow_html=True)
                    st.write(df)

                    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 = st.columns(15)
                    with c1:
                        st.image(downloadbutton, output_format="PNG", channels="RGB", width=25)
                    with c2:
                        st.markdown(filedownload(df), unsafe_allow_html=True)
                    return df

                def build_model_clf(input_data):
                    download = './Utils/Pictures/Download-Icon.png'
                    downloadbutton = Image.open(download)

                    # Reads in saved regression model
                    cf_subc_ic50 = './Utils/Pages/Models/Classification/IC50/rf_maccs_model.pkl'
                    load_model_cf = pickle.load(open(cf_subc_ic50, 'rb'))

                    # Apply model to make predictions
                    prediction_clf = load_model_cf.predict(input_data)

                    st.markdown('###### **$IC_{50}$**')
                    prediction_output_clf = pd.Series(prediction_clf, name='Classification')
                    molecule_name = pd.Series(compound_name, name='Compound Name/ID')

                    df = pd.concat([molecule_name, prediction_output_clf], axis=1)

                    st.markdown(
                        "The compound, " + str(compound_name) + " is " +
                        "<span style='color:green'><b>" + str(prediction_output_clf[0]) + "</b></span>" +
                        " against the target.", unsafe_allow_html=True)
                    st.write(df)

                    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 = st.columns(15)
                    with c1:
                        st.image(downloadbutton, output_format="PNG", channels="RGB", width=25)
                    with c2:
                        st.markdown(filedownload(df), unsafe_allow_html=True)
                    return df

                with st.spinner("PPGBioPred is calculating Bioactivity..."):
                    desc_calc()

                    # Read in calculated descriptors and display the dataframe
                    desc_reg = pd.read_csv('descriptors_output_pic50_RF.csv')
                    desc_clf = pd.read_csv('descriptors_output_pic50_CF.csv')
                    # Read descriptor list used in previously built model
                    Xlist_clf = list(pd.read_csv('./Utils/Pages/Models/Classification/IC50/df_MACCS_final_classification.csv').columns)
                    Xlist_reg = list(pd.read_csv('./Utils/Pages/Models/Regression/IC50/df_Substructure_final.csv').columns)
                    desc_subset_clf = desc_clf[Xlist_clf]
                    desc_subset_reg = desc_reg[Xlist_reg]
                    # Apply trained model to make prediction on query compounds
                    build_model_reg(desc_subset_reg)
                    build_model_clf(desc_subset_clf)

