import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc


st.title("Welcome to the Credit Score Classification Model ")


data = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

model = joblib.load("model.joblib")

Age = st.number_input("Enter age:", 18, 80, step=1, format="%d")
Occupation = st.selectbox("Select Occupation", data['Occupation'].unique().tolist())
Log_Annual_Income = st.number_input("Enter annual income", step=1000, format="%d")
Log_Monthly_Inhand_Salary = st.number_input("Enter monthly inhand salary", step=1000, format="%d")
Num_Bank_Accounts = st.selectbox('Select Number of bank accounts', data['Num_Bank_Accounts'].unique().tolist())
Num_Credit_Card = st.selectbox("Select number of credit cards", data['Num_Credit_Card'].unique().tolist())
Interest_Rate = st.number_input("Enter average interest rate of all loans", step=0.01, format="%.2f")
Num_of_Loan = st.selectbox("Select number of loans", data['Num_of_Loan'].unique().tolist())
Delay_from_due_date = st.number_input("Enter average number of days delay for repayments?", step=1, format='%d')
Num_of_Delayed_Payment = st.number_input("On an average how much times delayed for payments?", step=1, format='%d')
Changed_Credit_Limit = st.number_input("How much percent credit limit has changed?", 0.0,100.0, step=0.01, format='%.2f')
Num_Credit_Inquiries = st.number_input("How much times inquired credit score?", step=1, format='%d')
Credit_Mix = st.selectbox("How is credit mix", data['Credit_Mix'].unique().tolist())
Outstanding_Debt = st.number_input("How much outstanding debts have?", step=1000.0, format='%.2f')
Credit_Utilization_Ratio = st.number_input("What is credit utilization ratio", step=1.0, format='%.2f')
Credit_History_Age = st.number_input("What is credit history age", step=1.0, format='%.2f')
Payment_of_Min_Amount = st.selectbox("Any minimum payment due?",data['Payment_of_Min_Amount'].unique().tolist())
Payment_Behaviour = st.selectbox("Select payment behaviour", data['Payment_Behaviour'].unique().tolist())
Monthly_Balance = st.number_input("What is monthly balance", step=100.0, format='%.2f')
Auto_Loan = st.selectbox("Is there an auto loan?",data['Auto_Loan'].unique().tolist())
Personal_Loan = st.selectbox("Is there a personal loan?",data['Personal_Loan'].unique().tolist())
Credit_builder_Loan = st.selectbox("Is there a Credit-builder Loan?",data['Credit-builder_Loan'].unique().tolist())
Home_Equity_Loan = st.selectbox("Is there a home equity loan?",data['Home_Equity_Loan'].unique().tolist())
Mortgage_Loan = st.selectbox("Is there a mortgage loan?",data['Mortgage_Loan'].unique().tolist())
Student_Loan = st.selectbox("Is there a student loan?",data['Student_Loan'].unique().tolist())
Debt_Consolidation_Loan = st.selectbox("Is there a debt consolidation loan?",data['Debt_Consolidation_Loan'].unique().tolist())
Payday_Loan = st.selectbox("Is there a payday loan?",data['Payday_Loan'].unique().tolist())
Not_Specified_Loan = st.selectbox("Is there a loan that is not specified?",data['Not_Specified_Loan'].unique().tolist())
No_Data_of_Loan = st.selectbox("Is there any data on loans?",data['No_Data_of_Loan'].unique().tolist())
Log_Total_EMI_per_month = st.number_input("Enter the total EMI per month", step=1, format='%d')
Log_Amount_invested_monthly = st.number_input("Enter the total amount invested monthly", step=1, format='%d')

input_data = pd.DataFrame([[Age, Occupation, Num_Bank_Accounts, Num_Credit_Card, Interest_Rate, Num_of_Loan, Delay_from_due_date,
                            Num_of_Delayed_Payment, Changed_Credit_Limit, Num_Credit_Inquiries, Credit_Mix, Outstanding_Debt,
                            Credit_Utilization_Ratio, Credit_History_Age, Payment_of_Min_Amount, Payment_Behaviour, Monthly_Balance,
                            Log_Annual_Income, Log_Monthly_Inhand_Salary, Auto_Loan, Personal_Loan, Credit_builder_Loan, Home_Equity_Loan,
                            Mortgage_Loan, Student_Loan, Debt_Consolidation_Loan, Payday_Loan, Not_Specified_Loan, No_Data_of_Loan,
                            Log_Total_EMI_per_month, Log_Amount_invested_monthly]],
                            columns = ['Age','Occupation','Num_Bank_Accounts','Num_Credit_Card','Interest_Rate','Num_of_Loan','Delay_from_due_date',
                            'Num_of_Delayed_Payment','Changed_Credit_Limit','Num_Credit_Inquiries','Credit_Mix','Outstanding_Debt',
                            'Credit_Utilization_Ratio','Credit_History_Age','Payment_of_Min_Amount','Payment_Behaviour','Monthly_Balance',
                            'Log_Annual_Income','Log_Monthly_Inhand_Salary','Auto_Loan','Personal_Loan','Credit-builder_Loan','Home_Equity_Loan',
                            'Mortgage_Loan','Student_Loan','Debt_Consolidation_Loan','Payday_Loan','Not_Specified_Loan','No_Data_of_Loan',
                            'Log_Total_EMI_per_month','Log_Amount_invested_monthly'])

st.dataframe(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    # if prediction == 1:
    #     st.success("This customer has a good credit score")
    # else:
    #     st.success("This customer’s credit score is below the ideal range")

    if prediction == 1:
        bg_color = "#E9FCE9"  
        text = "This customer has a good credit score"
        text_color = "#002147"
    else: 
        bg_color = "#F8D3D3"  
        text = "This customer has a bad credit score"
        text_color = "#8B0000"  

    st.markdown(
    f"""
    <div style="background-color:{bg_color}; padding:5px 10px; border-radius:5px; width:100%; 
                display:inline-block; line-height:1.5;">
        <span style="color:{text_color}; font-size:16px; font-weight:bold;">{text}</span>
    </div>
    """,
    unsafe_allow_html=True
)

    
tab1 ,tab2, tab3, tab4, tab5 = st.tabs(['ROC AUC score','F1 Score','Precision Score','Recall Score', 'Accurecy Score'])

with tab1:
    st.write("ROC AUC score of the model on the test dataset")
    roc_auc = 0.9353

    st.markdown(
    f"""
    <div style="background-color:#E9FCE9; padding:5px; border-radius:5px; width:200px; min-height:20px; text-align:center;">
        <h4 style="color:#002147; font-size:16px; margin:5px;">ROC AUC score: {roc_auc}</h4>
    </div>
    """,
    unsafe_allow_html=True
)


with tab2:
    st.write("F1 score of the model on the test dataset")
    
    f1_score = 0.9169
    st.markdown(
    f"""
    <div style="background-color:#E9FCE9; padding:5px; border-radius:5px; width:200px; min-height:20px; text-align:center;">
        <h4 style="color:#002147; font-size:16px; margin:5px;">f1 score: {f1_score}</h4>
    </div>
    """,
    unsafe_allow_html=True
)
    

with tab3:
    st.write("Precision of the model on the test dataset")
   
    Precision = 0.9250
    st.markdown(
    f"""
    <div style="background-color:#E9FCE9; padding:5px; border-radius:5px; width:200px; min-height:20px; text-align:center;">
        <h4 style="color:#002147; font-size:16px; margin:5px;">Precision score: {Precision}</h4>
    </div>
    """,
    unsafe_allow_html=True
)

with tab4:
    st.write("Recall of the model on the test dataset")
   
    Recall = 0.9088
    st.markdown(
    f"""
    <div style="background-color:#E9FCE9; padding:5px; border-radius:5px; width:200px; min-height:20px; text-align:center;">
        <h4 style="color:#002147; font-size:16px; margin:5px;">Recall score: {Recall}</h4>
    </div>
    """,
    unsafe_allow_html=True
)

with tab5:
    st.write("Accuracy of the model on the test dataset")
  
    Accuracy = 0.8836
    st.markdown(
    f"""
    <div style="background-color:#E9FCE9; padding:5px; border-radius:5px; width:200px; min-height:20px; text-align:center;">
        <h4 style="color:#002147; font-size:16px; margin:5px;">Accuracy score: {Accuracy}</h4>
    </div>
    """,
    unsafe_allow_html=True
)


