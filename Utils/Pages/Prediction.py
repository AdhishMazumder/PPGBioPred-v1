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
                    bashCommand = (
                        "java -Xms2G -Xmx2G -Djava.awt.headless=true "
                        "-jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar "
                        "-removesalt -standardizenitro -fingerprints "
                        "-descriptortypes ./PaDEL-Descriptor/SubstructureFingerprintCount.xml "
                        "-dir ./ -file descriptors_output_pic50_RF.csv"
                    )

                    try:
                        process = subprocess.Popen(bashCommand, shell=True, stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE)
                        output, error = process.communicate()

                        if process.returncode == 0:
                            print("Descriptor calculation completed successfully.")
                        else:
                            print(f"Error encountered:\n{error.decode('utf-8')}")
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

                def build_model(input_data):
                    download = './Utils/Pictures/Download-Icon.png'
                    downloadbutton = Image.open(download)

                    # Reads in saved regression model
                    rf_subc_ic50 = './Utils/Pages/Models/rf_pparg_subc_model_IC50.pkl'
                    load_model = pickle.load(open(rf_subc_ic50, 'rb'))
                    cf_subc_ic50 = './Utils/Pages/Models/cf_pparg_subc_model_IC50.pkl'
                    load_model_cf = pickle.load(open(cf_subc_ic50, 'rb'))

                    # Apply model to make predictions
                    prediction = load_model.predict(input_data)
                    prediction_cf = load_model_cf.predict(input_data)

                    st.markdown('###### **$IC_{50}$**')
                    prediction_output = pd.Series(prediction, name='pIC50')
                    prediction_output_cf = pd.Series(prediction_cf, name='Classification')
                    molecule_name = pd.Series(compound_name, name='Compound Name/ID')

                    # Convert pIC50 to IC50 (in M) and then to nM
                    calc_IC50 = 10 ** -prediction_output * 10 ** 9
                    prediction_IC50 = pd.Series(calc_IC50, name='IC50 (nM)')

                    df = pd.concat([molecule_name, prediction_output, prediction_IC50, prediction_output_cf], axis=1)

                    st.markdown(
                        "The **$IC_{50}$** of " + str(compound_name) + " is " + "<span style='color:green'><b>" +
                        str(np.round(calc_IC50[0], 2)) + " nM" + "</b></span>" + " and it is " +
                        "<span style='color:green'><b>" + str(prediction_output_cf[0]) + "</b></span>" +
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
                    desc = pd.read_csv('descriptors_output_pic50_RF.csv')
                    # Read descriptor list used in previously built model
                    Xlist = list(pd.read_csv('SUBC_IC50.csv').columns)
                    desc_subset = desc[Xlist]
                    # Apply trained model to make prediction on query compounds
                    build_model(desc_subset)

    # Target Specification for Pathways
    if compound_pathway == "EC50":
        wnt_targets = col2.selectbox(
            'Select the target',
            ('PPARγ',), index=0)
        st.markdown("---")

        # Button functionality
        if st.button("Submit for Prediction", key='Submit1'):
            submission_data = pd.DataFrame(
                {"SMILES": [compound_smiles], "Compound Name/ID": [compound_name]})
            submission_data.to_csv('molecule.smi', sep='\t', header=False,
                                   index=False)
            if wnt_targets == "PPARγ":
                st.markdown('## **Prediction Results**')

                # Molecular descriptor calculator
                def desc_calc():
                    # Specify the full command as a single string
                    bashCommand = (
                        "java -Xms2G -Xmx2G -Djava.awt.headless=true "
                        "-jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar "
                        "-removesalt -standardizenitro -fingerprints "
                        "-descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml "
                        "-dir ./ -file descriptors_output_pec50.csv"
                    )

                    try:
                        process = subprocess.Popen(bashCommand, shell=True, stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE)
                        output, error = process.communicate()
                        # os.remove('molecule.smi')

                        if process.returncode == 0:
                            print("Descriptor calculation completed successfully.")
                        else:
                            print(f"Error encountered:\n{error.decode('utf-8')}")
                    except Exception as e:
                        print(f"An error occurred: {e}")

                # File download
                def filedownload(df):
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(
                        csv.encode()).decode()  # strings <-> bytes conversions
                    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download </a>'
                    return href

                # Model building
                def build_model(input_data):
                    download = './Utils/Pictures/Download-Icon.png'
                    downloadbutton = Image.open(download)
                    # Reads in saved regression model
                    rf_subc_ec50 = './Utils/Pages/Models/rf_pparg_pub_model_EC50.pkl'
                    load_model = pickle.load(open(rf_subc_ec50, 'rb'))
                    cf_subc_ec50 = './Utils/Pages/Models/cf_pparg_pub_model_EC50.pkl'
                    load_model_cf = pickle.load(open(cf_subc_ec50, 'rb'))

                    # Apply model to make predictions
                    prediction = load_model.predict(input_data)
                    prediction_cf = load_model_cf.predict(input_data)

                    st.markdown('###### **$EC_{50}$**')
                    prediction_output = pd.Series(prediction, name='pEC50')
                    prediction_output_cf = pd.Series(prediction_cf, name='Classification')
                    molecule_name = pd.Series(compound_name,
                                              name='Compound Name/ID')
                    calc_EC50 = np.power(10, 9) * np.power(10, -prediction)
                    prediction_EC50 = pd.Series(calc_EC50, name='EC50 (nM)')
                    df = pd.concat(
                        [molecule_name, prediction_output, prediction_EC50, prediction_output_cf], axis=1)

                    st.markdown("The **$EC_{50}$** of " + str(compound_name) + " is " + "<span style='color:green'><b>" +
                                str(np.round(calc_EC50[0], 2)) + " nM" + "</b></span>" + " and it is " +
                                "<span style='color:green'><b>" + str(prediction_output_cf[0]) + "</b></span>" +
                                " against the target.", unsafe_allow_html=True)
                    st.write(df)

                    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 = st.columns(
                        15)
                    with c1:
                        st.image(downloadbutton, output_format="PNG",
                                 channels="RGB", width=25)
                    with c2:
                        st.markdown(filedownload(df), unsafe_allow_html=True)
                    return df

                with st.spinner("PPGBioPred is calculating Bioactivity..."):
                    desc_calc()

                    # Read in calculated descriptors and display the dataframe
                    desc = pd.read_csv('descriptor_output_pec50.csv')
                    # Read descriptor list used in previously built model
                    Xlist = list(pd.read_csv('Pub_EC50.csv').columns)
                    desc_subset = desc[Xlist]
                    # Apply trained model to make prediction on query compounds
                    build_model(desc_subset)

    # Target Specification for Pathways
    if compound_pathway == "Activity %":
        wnt_targets = col2.selectbox(
            'Select the target',
            ('PPARγ',), index=0)
        st.markdown("---")

        # Button functionality
        if st.button("Submit for Prediction", key='Submit2'):
            submission_data = pd.DataFrame(
                {"SMILES": [compound_smiles], "Compound Name/ID": [compound_name]})
            submission_data.to_csv('molecule.smi', sep='\t', header=False,
                                   index=False)
            if wnt_targets == "PPARγ":
                st.markdown('## **Prediction Result**')

                # Molecular descriptor calculator
                def desc_calc():
                    # Specify the full command as a single string
                    bashCommand = (
                        "java -Xms2G -Xmx2G -Djava.awt.headless=true "
                        "-jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar "
                        "-removesalt -standardizenitro -fingerprints "
                        "-descriptortypes ./PaDEL-Descriptor/MACCSFingerprinter.xml "
                        "-dir ./ -file descriptors_output_Activity.csv"
                    )

                    try:
                        process = subprocess.Popen(bashCommand, shell=True, stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE)
                        output, error = process.communicate()

                        if process.returncode == 0:
                            print("Descriptor calculation completed successfully.")
                        else:
                            print(f"Error encountered:\n{error.decode('utf-8')}")
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

                # Model building
                def build_model(input_data):
                    download = './Utils/Pictures/Download-Icon.png'
                    downloadbutton = Image.open(download)
                    # Reads in saved regression model
                    rf_maccs_activity = './Utils/Pages/Models/rf_pparg_maccs_model_activity.pkl'
                    load_model = pickle.load(open(rf_maccs_activity, 'rb'))
                    cf_maccs_activity = './Utils/Pages/Models/cf_pparg_maccs_model_activity.pkl'
                    load_model_cf = pickle.load(open(cf_maccs_activity, 'rb'))

                    # Apply model to make predictions
                    prediction = load_model.predict(input_data)
                    prediction_cf = load_model_cf.predict(input_data)

                    st.markdown('###### **Activity (%)**')
                    activity_calc = np.multiply(prediction, 100)
                    prediction_output = pd.Series(activity_calc, name='Activity %')
                    prediction_output_cf = pd.Series(prediction_cf, name='Classification')
                    molecule_name = pd.Series(compound_name, name='Compound Name/ID')
                    df = pd.concat([molecule_name, prediction_output, prediction_output_cf], axis=1)
                    st.markdown("The **activity** of " + str(compound_name) + " is " + "<span style='color:green'><b>" +
                                str(np.round(activity_calc[0], 2)) + "</b></span>" + "% and it is " +
                                "<span style='color:green'><b>" + str(prediction_output_cf[0]) + "</b></span>" +
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
                    desc = pd.read_csv('descriptor_output_Activity.csv')
                    # Read descriptor list used in previously built model
                    Xlist = list(
                        pd.read_csv('MACCS_Activity.csv').columns)
                    desc_subset = desc[Xlist]
                    # Apply trained model to make prediction on query compounds
                    build_model(desc_subset)
