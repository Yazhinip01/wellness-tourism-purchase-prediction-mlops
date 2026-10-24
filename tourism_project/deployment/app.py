import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_wellness_tourism_model.joblib")
model = joblib.load(model_path)

st.title("Wellness Tourism Purchase Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the newly
launched Wellness Tourism Package based on their demographic details,
travel preferences, and interaction with the sales team.

Enter the customer's details below to get a prediction and help the tourism
company identify potential customers for targeted marketing.
""")

type_of_contact         = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
Occupation              = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
city_tier               = st.selectbox("City Tier", ["1", "2", "3"])
Gender                  = st.selectbox("Gender", ["Male", "Female"])
product_pitched         = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
marital_status          = st.selectbox("Marital Status", ["Married", "Unmarried", "Divorced"])
preferred_property_star = st.selectbox("Preferred Property Star", ["1", "2", "3", "4", "5"])
passport                = st.selectbox("Has a Passport?", [0, 1])
ownCar                  = st.selectbox("Owns a Car?", [0, 1])
no_of_personViditing    = st.number_input("Number of Person Visiting", min_value = 1, max_value = 20,step = 1)
No_of_trips             = st.number_input("Number of trips", min_value= 0, max_value = 50)
No_of_children_visiting = st.number_input("Number of children visiting", min_value=0, max_value = 15)
duration_of_pitch       = st.number_input("Duration of Pitch", min_value = 0.0, step =1.0)
monthly_income          = st.number_input("Monthly Income", min_value = 0.0)
age                     = st.number_input("Age", min_value = 0, max_value = 100)
no_of_followups         = st.number_input("Number of Followups", min_value = 0, max_value = 20)
pitch_satisfaction_score  = st.number_input("Pitch Satisfaction Score", min_value = 1, max_value = 5)
designation               =st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

input_data = pd.DataFrame([{
    "TypeofContact": type_of_contact,
    "Occupation": Occupation,
    "CityTier": city_tier,
    "Gender": Gender,
    "ProductPitched": product_pitched ,
    "MaritalStatus": marital_status,
    "PreferredPropertyStar": preferred_property_star,
    "NumberOfPersonVisiting": no_of_personViditing,
    "NumberOfTrips": No_of_trips,
    "NumberOfChildrenVisiting": No_of_children_visiting,
    "DurationOfPitch": duration_of_pitch,
    "MonthlyIncome": monthly_income,
    "Passport": passport,
    "OwnCar": ownCar,
    "Age": age,
    "NumberOfFollowups": no_of_followups,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "Designation": designation
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "likely to Purchase the Wellness Tourism Package" if prediction == 1 else "UnLikely to Purchase the Wellness Tourism Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
