import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

# Load the pre-trained model
# Load the model
model = joblib.load("model.pkl")


# Function to preprocess user input
def preprocess_input(input_data):
    # Placeholder for preprocessing steps
    # Convert categorical variables to numeric using one-hot encoding
    mjob_mapping = {"at_home": 0, "other": 1, "services": 2, "teacher": 3, "health": 4}
    fjob_mapping = {"at_home": 0, "other": 1, "services": 2, "health": 3, "teacher": 4}
    reason_mapping = {"course": 0, "home": 1, "reputation": 2, "other": 3}
    guardian_mapping = {"other": 0, "mother": 1, "father": 2}

    processed_input = {
        "school": 1 if input_data["school"] == "GP" else 0,
        "sex": 1 if input_data["sex"] == "F" else 0,
        "age": input_data["age"],
        "address": 1 if input_data["address"] == "U" else 0,
        "famsize": 1 if input_data["famsize"] == "LE3" else 0,
        "Pstatus": 1 if input_data["Pstatus"] == "A" else 0,
        "Medu": input_data["Medu"],
        "Fedu": input_data["Fedu"],
        "Mjob": mjob_mapping.get(input_data["Mjob"], 1),  # Use 1 as default for 'other'
        "Fjob": fjob_mapping.get(input_data["Fjob"], 1),  # Use 1 as default for 'other'
        "reason": reason_mapping.get(
            input_data["reason"], 3
        ),  # Use 3 as default for 'other'
        "guardian": guardian_mapping.get(
            input_data["guardian"], 0
        ),  # Use 0 as default for 'other'
        "traveltime": input_data["traveltime"],
        "studytime": input_data["studytime"],
        "failures": input_data["failures"],
        "schoolsup": 1 if input_data["schoolsup"] == "yes" else 0,
        "famsup": 1 if input_data["famsup"] == "yes" else 0,
        "paid": 1 if input_data["paid"] == "yes" else 0,
        "activities": 1 if input_data["activities"] == "yes" else 0,
        "nursery": 1 if input_data["nursery"] == "yes" else 0,
        "higher": 1 if input_data["higher"] == "yes" else 0,
        "internet": 1 if input_data["internet"] == "yes" else 0,
        "romantic": 1 if input_data["romantic"] == "yes" else 0,
        "famrel": input_data["famrel"],
        "freetime": input_data["freetime"],
        "goout": input_data["goout"],
        "Dalc": input_data["Dalc"],
        "Walc": input_data["Walc"],
        "health": input_data["health"],
        "absences": input_data["absences"],
        "G1": input_data["G1"],
        "G2": input_data["G2"],
    }
    # Return only the features used for prediction
    return {
        key: value
        for key, value in processed_input.items()
        if key
        in [
            "school",
            "sex",
            "age",
            "address",
            "famsize",
            "Pstatus",
            "Medu",
            "Fedu",
            "Mjob",
            "Fjob",
            "reason",
            "guardian",
            "traveltime",
            "studytime",
            "failures",
            "schoolsup",
            "famsup",
            "paid",
            "activities",
            "nursery",
            "higher",
            "internet",
            "romantic",
            "famrel",
            "freetime",
            "goout",
            "Dalc",
            "Walc",
            "health",
            "absences",
            "G1",
            "G2",
        ]
    }


# Function to predict G3
def predict_G3(input_data):
    # Preprocess input
    processed_input = preprocess_input(input_data)
    # Convert input to DataFrame
    input_df = pd.DataFrame([processed_input])
    # Make prediction
    prediction = model.predict(input_df)
    # Round and clip prediction to be between 0 and 20
    rounded_prediction = np.clip(np.round(prediction), 0, 20)
    return rounded_prediction[0]


# Streamlit UI
st.title("Student Performance Prediction")

# Collect user input
# Demographic Information
st.header("Demographic Information")
school = st.selectbox("School:", ["GP", "MS"], help="Student's school")
sex = st.selectbox("Gender:", ["F", "M"], help="Student's gender")
age = st.number_input("Age:", min_value=15, max_value=22, help="Student's age")
address = st.selectbox("Address Type:", ["U", "R"], help="Student's home address type")
famsize = st.selectbox("Family Size:", ["LE3", "GT3"], help="Family size")
Pstatus = st.selectbox(
    "Parent Cohabitation Status:", ["T", "A"], help="Parent's cohabitation status"
)
Medu = st.slider("Mother's Education Level:", 0, 4, help="Mother's education level")
Fedu = st.slider("Father's Education Level:", 0, 4, help="Father's education level")

# Occupation Information
st.header("Occupation Information")
Mjob = st.selectbox(
    "Mother's Occupation:",
    ["teacher", "health", "services", "at_home", "other"],
    help="Mother's job",
)
Fjob = st.selectbox(
    "Father's Occupation:",
    ["teacher", "health", "services", "at_home", "other"],
    help="Father's job",
)

# Reasons and Support
st.header("Reasons and Support")
reason = st.selectbox(
    "Reason for Choosing School:",
    ["home", "reputation", "course", "other"],
    help="Reason to choose this school",
)
guardian = st.selectbox(
    "Guardian:", ["mother", "father", "other"], help="Student's guardian"
)
traveltime = st.slider(
    "Travel Time to School:", 1, 4, help="Home to school travel time"
)
studytime = st.slider("Weekly Study Time:", 1, 4, help="Weekly study time")
failures = st.slider("Past Class Failures:", 0, 3, help="Number of past class failures")

# Support and Activities
st.header("Support and Activities")
schoolsup = st.selectbox(
    "Extra Educational Support:", ["yes", "no"], help="Extra educational support"
)
famsup = st.selectbox(
    "Family Educational Support:", ["yes", "no"], help="Family educational support"
)
paid = st.selectbox(
    "Extra Paid Classes:",
    ["yes", "no"],
    help="Extra paid classes within the course subject",
)
activities = st.selectbox(
    "Extra-curricular Activities:", ["yes", "no"], help="Extra-curricular activities"
)
nursery = st.selectbox(
    "Attended Nursery School:", ["yes", "no"], help="Attended nursery school"
)

# Aspirations and Lifestyle
st.header("Aspirations and Lifestyle")
higher = st.selectbox(
    "Wants to Take Higher Education:",
    ["yes", "no"],
    help="Wants to take higher education",
)
internet = st.selectbox(
    "Internet Access at Home:", ["yes", "no"], help="Internet access at home"
)
romantic = st.selectbox(
    "In a Romantic Relationship:", ["yes", "no"], help="In a romantic relationship"
)
famrel = st.slider(
    "Quality of Family Relationships:", 1, 5, help="Quality of family relationships"
)
freetime = st.slider("Free Time After School:", 1, 5, help="Free time after school")
goout = st.slider("Going Out with Friends:", 1, 5, help="Going out with friends")
Dalc = st.slider(
    "Workday Alcohol Consumption:", 1, 5, help="Workday alcohol consumption"
)
Walc = st.slider(
    "Weekend Alcohol Consumption:", 1, 5, help="Weekend alcohol consumption"
)
health = st.slider("Current Health Status:", 1, 5, help="Current health status")
absences = st.number_input(
    "Number of School Absences:",
    min_value=0,
    max_value=93,
    help="Number of school absences",
)

# Grading Information
st.header("Grading Information")
G1 = st.number_input(
    "First Period Grade:", min_value=0, max_value=20, help="First period grade"
)
G2 = st.number_input(
    "Second Period Grade:", min_value=0, max_value=20, help="Second period grade"
)

# Button for Prediction
if st.button("Predict Final Grade"):
    user_input = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "G1": G1,
        "G2": G2,
    }
    predicted_G3 = predict_G3(user_input)
    st.write("Predicted Final Grade (G3):", predicted_G3)
