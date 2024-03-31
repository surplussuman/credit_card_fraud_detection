import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st


dataset= pd.read_csv("creditcard.csv")

correct = dataset[dataset.Class==0]
fraud = dataset[dataset.Class==1]

sucess_sample = correct.sample(n=492)
dataset2 = pd.concat([sucess_sample,fraud],axis=0)

x = dataset2.drop('Class',axis=1)
y = dataset2['Class']
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, stratify=y,random_state=2)

model = LogisticRegression()
model.fit(X_train,Y_train)

train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test),Y_test)

st.title("Credit Card fraud Detection")
st.write("Enter the following features to check if the transaction if legitimate or fradulent")

input_df = st.text_input("Enter all features")
input_df_lst= input_df.split(',')

submit = st.button("Submit")

if submit:
    features = np.array(input_df_lst, dtype=np.float64)
    prediction = model.predict(features.reshape(1,-1))

    if prediction[0] == 0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fraudulent Transaction")
