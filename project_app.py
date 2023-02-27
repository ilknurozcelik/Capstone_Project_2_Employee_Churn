
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from sklearn.linear_model import Lasso
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
import pickle

# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://miro.medium.com/max/420/1*cVrhSuLkIbEGjIRi3JODcQ.jpeg");
#              background-attachment: fixed;
#              background-size: cover;
#              background-opacity: 0.5;
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )

# add_bg_from_url()

# original_title = '''<p style="font-family:Courier; color:Blue; font-size: 20px;">Original Title</p>'''
# st.markdown(original_title, unsafe_allow_html=True)
# # st.image(image, channels="BGR")

# new_title = '''<p style="font-family:sans-serif; color:Green; font-size: 42px;">New image</p>'''
# st.markdown(new_title, unsafe_allow_html=True)
# st.image(image, channels="BGR")



st.markdown("<h1 style='text-align:center; font-family:verdana; font-size:250%; opacity:1; color:black; '><b>Employee Churn Prediction<b></h2>", unsafe_allow_html=True)
df =  pd.read_csv("df_without_dp.csv")

# Adding image
st.image("https://miro.medium.com/max/420/1*cVrhSuLkIbEGjIRi3JODcQ.jpeg", width=700)


# add button
show_data = '<p style="font-family:verdana; color:red; font-size: 20px;"><b>Show Data</b></p>'
st.markdown(show_data, unsafe_allow_html=True)
if st.checkbox(" ") :
    st.table(df.head())
    
# add warning
warning = '<p style="font-family:verdana; color:black; font-size: 20px;"> <b>Please input the features of employee using sidebar, before making churn prediction!!!</b></p>'
st.markdown(warning, unsafe_allow_html=True)


st.sidebar.title("Please select features of employee")

# Collects user input features into dataframe
def user_input_features() :
    satisfaction_level  = st.sidebar.slider("Satisfation Level", df["satisfaction_level"].min(),
                                            df["satisfaction_level"].max(),
                                            df["satisfaction_level"].mean())
    last_evaluation = st.sidebar.slider("Last Evaluation", df["last_evaluation"].min(),
                                            df["last_evaluation"].max(),
                                            df["last_evaluation"].mean())
    number_project = st.sidebar.selectbox("The Number of Projects",df["number_project"].unique())
    average_montly_hours = st.sidebar.number_input("Average Monthly Hours",min_value=df["average_montly_hours"].min(),
                                                   max_value=df["average_montly_hours"].max())
    time_spend_company = st.sidebar.selectbox("Experience (Years)", df["time_spend_company"].unique())
    
    
#     work_accident = st.sidebar.radio("Has work accident?",('Yes', 'No'))
#     if work_accident == 'Yes':
#         work_accident = 1       
#     else:
#         work_accident = 0
    
     
#     promotion_last_5years = st.sidebar.radio("Has promotion last five years?",('Yes', 'No'))
#     if promotion_last_5years == 'Yes':
#         promotion_last_5years = 1       
#     else:
#         promotion_last_5years = 0
    
#     departments = st.sidebar.selectbox("Departments",df["departments"].unique())
#     salary = st.sidebar.selectbox("Salary",df["salary"].unique())
      
    
    data = {"satisfaction_level" : satisfaction_level,
            "last_evaluation" : last_evaluation,
            "number_project" : number_project,
            "average_montly_hours" : average_montly_hours,
            "time_spend_company" : time_spend_company,
#             "work_accident" : work_accident,
#            "promotion_last_5years" : promotion_last_5years,
#            "departments" : departments,
#            "salary" : salary
           }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Read the saved model
model= pickle.load(open("final_model_employee_churn", "rb"))


# Apply model to make predictions

if st.button('Predict Employee Churn'):
    
    if model.predict(input_df)[0] == 1:
        st.error(f'Left Employee')
               
    else :
        st.success(f'Ongoing Employee')
    
    fig = go.Figure(go.Indicator(  # probability gauge chart)
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = model.predict_proba(input_df)[0][1],
        mode = "gauge+number+delta",
        title = {'text': "Churn Probability"},
        delta = {'reference': 0.5},
        gauge = {'axis': {'range': [None, 1]},
                 'steps' : [{'range': [0, 0.5], 'color': "red"},
                            {'range': [0.5, 1], 'color': "green"}],
                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.5}}))

    st.plotly_chart(fig, use_container_width=True)  # to display chart on application page

st.markdown("Thank you for visiting our **Employee Churn Prediction** page.")





