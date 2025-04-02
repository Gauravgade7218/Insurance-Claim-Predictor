import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Insurance Claim Prediction App")
st.write("Enter the details below to predict claim approval.")

# User Inputs
CLMSEX = st.selectbox("Claimant Sex", [0, 1])  # 0 = Female, 1 = Male
CLMINSUR = st.selectbox("Claimant Insured", [0, 1])  # 0 = No, 1 = Yes
SEATBELT = st.selectbox("Seatbelt Used", [0, 1])  # 0 = No, 1 = Yes
CLMAGE = st.number_input("Claimant Age", min_value=0, max_value=120, step=1)
LOSS = st.number_input("Loss Amount", min_value=0.0, step=0.01)
CLAIM_AMOUNT_REQUESTED = st.number_input("Claim Amount Requested", min_value=0.0, step=0.01)
CLAIM_APPROVAL_STATUS = st.selectbox("Claim Approval Status", [0, 1])  # 0 = Rejected, 1 = Approved
SETTLEMENT_AMOUNT = st.number_input("Settlement Amount", min_value=0.0, step=0.01)

# Prepare input data
input_data = np.array([
    CLMSEX, CLMINSUR, SEATBELT, CLMAGE, LOSS,
    CLAIM_AMOUNT_REQUESTED, CLAIM_APPROVAL_STATUS, SETTLEMENT_AMOUNT
]).reshape(1, -1)

# Predict
if st.button("Predict Claim Outcome"):
    prediction = model.predict(input_data)[0]
    result = "Approved" if prediction == 1 else "Rejected"
    st.write(f"### Prediction: {result}")
