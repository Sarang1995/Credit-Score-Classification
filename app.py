import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


st.title("Welcome to the Credit Score Classification Model ")


data = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

model = joblib.load("model.joblib")

Age = st.number_input("Enter your age:", 18, 80, step=1, format="%d")
Occupation = st.selectbox("Select your Occupation", data['Occupation'].unique().tolist())
Log_Annual_Income = st.number_input("Enter your annual income", step=1000, format="%d")
Log_Monthly_Inhand_Salary = st.number_input("Enter your monthly inhand salary", step=1000, format="%d")
Num_Bank_Accounts = st.selectbox('How much bank accounts you have?', data['Num_Bank_Accounts'].unique().tolist())
Num_Credit_Card = st.selectbox("How much credit card you have?", data['Num_Credit_Card'].unique().tolist())
Interest_Rate = st.number_input("Enter your interest rate", step=0.01, format="%.2f")
Num_of_Loan = st.selectbox("How much loans you have taken?", data['Num_of_Loan'].unique().tolist())
Delay_from_due_date = st.number_input("How much days you delayed for repayment?", step=1, format='%d')
Num_of_Delayed_Payment = st.number_input("How much times you delayed your payment?", step=1, format='%d')
Changed_Credit_Limit = st.number_input("How much percent your credit limit has changed?", 0.0,100.0, step=0.01, format='%.2f')
Num_Credit_Inquiries = st.number_input("How much times you inquired your credit score?", step=1, format='%d')
Credit_Mix = st.selectbox("How is your credit mix", data['Credit_Mix'].unique().tolist())
Outstanding_Debt = st.number_input("How much outstanding debts you have?", step=1000.0, format='%.2f')
Credit_Utilization_Ratio = st.number_input("What is your credit utilization ratio", step=1.0, format='%.2f')
Credit_History_Age = st.number_input("What is your credit history age", step=1.0, format='%.2f')
Payment_of_Min_Amount = st.selectbox("You have any minimum payment due",data['Payment_of_Min_Amount'].unique().tolist())
Payment_Behaviour = st.selectbox("Select payment behaviour", data['Payment_Behaviour'].unique().tolist())
Monthly_Balance = st.number_input("What is monthly balance", step=100.0, format='%.2f')
Auto_Loan = st.selectbox("Does the customer have an auto loan?",data['Auto_Loan'].unique().tolist())
Personal_Loan = st.selectbox("Does the customer have an personal loan?",data['Personal_Loan'].unique().tolist())
Credit_builder_Loan = st.selectbox("Does the customer have an Credit-builder Loan?",data['Credit-builder_Loan'].unique().tolist())
Home_Equity_Loan = st.selectbox("Does the customer have an home equity loan?",data['Home_Equity_Loan'].unique().tolist())
Mortgage_Loan = st.selectbox("Does the customer have an mortgage loan?",data['Mortgage_Loan'].unique().tolist())
Student_Loan = st.selectbox("Does the customer have an student loan?",data['Student_Loan'].unique().tolist())
Debt_Consolidation_Loan = st.selectbox("Does the customer have an debt consolidation loan?",data['Debt_Consolidation_Loan'].unique().tolist())
Payday_Loan = st.selectbox("Does the customer have an payday loan?",data['Payday_Loan'].unique().tolist())
Not_Specified_Loan = st.selectbox("Does the customer not specified loan?",data['Not_Specified_Loan'].unique().tolist())
No_Data_of_Loan = st.selectbox("Do you not have any data of loans?",data['No_Data_of_Loan'].unique().tolist())
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

    if prediction == 1:
        st.success("This customer has a strong credit score")
    else:
        st.success("This customer’s credit score is below the ideal range")

y_test_binary = y_test["Credit_Score"] 
    
tab1 ,tab2, tab3, tab4, tab5 = st.tabs(['Accuracy Score','Precision Score','Recall Score','F1 Score', 'Confusion Matrix'])

with tab1:
    st.write("Accuracy of the model on the test dataset")
    with st.spinner("Loading accuracy... Please wait"):
        time.sleep(3)
        accurecy = accuracy_score(y_test_binary, model.predict(data))
    st.success(f"Accuracy score of model: {accurecy}")

with tab2:
    st.write("Precision of the model on the test dataset")
    with st.spinner("Loading Precision score... Please wait"):
        time.sleep(3)
        precision = precision_score(y_test_binary, model.predict(data))
    st.success(f"Precision score of model: {precision}")

with tab3:
    st.write("Recall of the model on the test dataset")
    with st.spinner("Loading Recall score... Please wait"):
        time.sleep(3)
        Recall = recall_score(y_test_binary, model.predict(data))
    st.success(f"Recall score of model: {Recall}")

with tab4:
    st.write("F1 score of the model on the test dataset")
    with st.spinner("Loading f1 score... Please wait"):
        time.sleep(3)
        f1 = f1_score(y_test_binary, model.predict(data))
    st.success(f"Recall score of model: {f1}")

with tab5:
    st.write("Confusion matrix of the model on the test dataset")
    with st.spinner("Loading confusion matrix... Please wait"):
        time.sleep(3)
        plt.figure(figsize=(2, 1))
        sns.heatmap(confusion_matrix(y_test_binary, model.predict(data)), annot=True, fmt="0.0f",annot_kws={"size": 8})
    st.pyplot(plt)