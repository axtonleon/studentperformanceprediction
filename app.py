import streamlit as st
import pickle as pk
from sklearn.preprocessing import StandardScaler
import time
import os
cwd = os.getcwd()

# Save the model paths to a list
model1 = "xgb_model(1).sav"
model2 = "logreg_model(2).sav"
model3 = "svm_model(1).sav"
model4 = "xgb_model(1).sav"

models = [model1, model2, model3, model4]

import pandas as pd
# Mapping of output options to numbers
education_level_mapping = {
    "none": 0,
    "primary education": 1,
    "secondary education": 2,
    "higher education": 3
}

failure_mapping = {
    "very low": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "very high": 4
}

sex_mapping = {
    "Male": 1,
    "Female": 0
}

support_mapping = {
    "Yes": 1,
    "No": 0
}

course_mapping = {
    "Yes": 1,
    "No": 0
}
mjob_mapping = {
    'teacher': 0, 'other': 1, 'civil services': 2, 'health care related': 3,
}
study_time_mapping = {
    "less than 2 hours": 1, "2 to 5 hours":2,"Greater than 10 hours":3
}
reason_mapping = {
    'course preference': 0, 'other': 1, 'close to home': 2, 'school reputation': 3
}
st.title("STUDENT SUCCESS PREDICTOR")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.text_input('Enter your age: ')
    Medu = st.selectbox('Enter your mother\'s education level: ',
                        ("none", "primary education", "secondary education", "higher education"))
    Fedu = st.selectbox('Enter your father\'s education level: ',
                        ("none", "primary education", "secondary education", "higher education"))
    
    Medu = education_level_mapping[Medu]
    Fedu = education_level_mapping[Fedu]
with col2:
    failures = st.selectbox('How often do you fail?: ', ("very low", "low", "moderate", "high", "very high"))
    sex_M = st.selectbox('Enter your sex: ', ("Male", "Female"))
    reason = st.selectbox('Your reason for attending Your school of choice: ', ('close to home', 'school reputation', 'course preference','other'))
    reason = reason_mapping[reason]
    failures = failure_mapping[failures]
    sex_M = sex_mapping[sex_M]
with col3:
    study_time = st.selectbox('How often do you study per day: ', ("less than 2 hours", "2 to 5 hours","Greater than 10 hours" ))
    higher_yes = st.selectbox('Do you like your course of study: ', ("Yes", "No"))
    Mjob = st.selectbox('Enter your school sponsors profession: ',
                        ("teacher", "health care related", "civil services", "other"))
    Mjob = mjob_mapping[Mjob]
    study_time = study_time_mapping[study_time]
    higher_yes = course_mapping[higher_yes]

model_mapping = {
    "Xgboost": 0,
    "Logistic Regression": 1,
    "Support vector Machine": 2,
    "Deep Neural Network": 3
}
selected_model = st.selectbox('Select the model:', ("Xgboost", "Logistic Regression", "Support vector Machine","Deep Neural Network"))
selected_model = model_mapping[selected_model]
model_path = models[selected_model]

if st.button('Predict'):
    # dictionary with list objects in values
    dets = {
        'age': [int(age)],
        'Medu': [Medu],
        'Fedu': [Fedu],
        'Mjob': [Mjob],
        'reason': [reason],
        'study_time': [study_time],
        'failures': [failures],
        'higher_yes': [higher_yes],
    }

    # creating a DataFrame object
    df = pd.DataFrame(dets)
    print(df.shape)

    loaded_model = pk.load(open(model_path, 'rb'))
    y_pred = loaded_model.predict(df)
    if model_path == r"C:\Users\Administrator\Downloads\tit2\svm_model(1).sav":
        if y_pred == 1:
            st.write("""
                We predict you are going to PASS.
                Probability of Success
                """)
        else:
            st.write("""
                We predict you are going to FAIL.
                Probability of Failure
                """)

    elif y_pred == 1:
        y_pred_proba = loaded_model.predict_proba(df)
        st.write("""
            We predict you are going to PASS.
            Probability of Success: {:.2f}
            """.format(y_pred_proba[0, 1]))
    else:
        y_pred_proba = loaded_model.predict_proba(df)
        st.write("""
            We predict you are going to FAIL.
            Probability of Failure: {:.2f}
            """.format(y_pred_proba[0, 0]))
