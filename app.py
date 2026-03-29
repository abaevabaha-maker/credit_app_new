import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.joblib")

st.title("Titanic Survival Prediction")

st.write("Введите данные пассажира:")

PassengerId = st.number_input("PassengerId")
Pclass = st.number_input("Pclass (1-3)")
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.number_input("Age")
SibSp = st.number_input("SibSp")
Parch = st.number_input("Parch")
Fare = st.number_input("Fare")
Embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# преобразование
Sex = 0 if Sex == "male" else 1
Embarked = {"S": 0, "C": 1, "Q": 2}[Embarked]

if st.button("Predict"):
    data = pd.DataFrame([[
        PassengerId, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    ]],
    columns=["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"])

    result = model.predict(data)

    if result[0] == 1:
        st.success("Выживет ✅")
    else:
        st.error("Не выживет ❌")

# Run streamlit in the background and expose it via localtunnel
!streamlit run app.py & npx localtunnel --port 8501
