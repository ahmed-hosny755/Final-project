import pickle as pkl
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# تحميل نموذج الانحدار من الملف
file= open("model.pkl", "rb")
model= pkl.load(file)
file.close()

# عنوان التطبيق
st.title('Diabetes Prediction Web Application')

# إنشاء واجهة تفاعلية
st.write("Enter values to predict the outcome:")

Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
Glucose = st.number_input('Glucose', min_value=0, max_value=200, value=0)
BloodPressure = st.number_input('BloodPressure', min_value=0, max_value=200, value=0)
SkinThickness = st.number_input('SkinThickness', min_value=0, max_value=100, value=0)
Insulin = st.number_input('Insulin', min_value=0, max_value=900, value=0)
BMI = st.number_input('BMI', min_value=0.0, max_value=70.0, value=0.0)
DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.0)
Age = st.number_input('Age', min_value=0, max_value=120, value=0)

# زر التنبؤ
if st.button('Predict'):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    prediction = model.predict(input_data)
    st.write(f"Prediction: {prediction[0]}")

    # يمكنك أيضًا تقييم النموذج وعرض متوسط الخطأ التربيعي إذا كان لديك مجموعة اختبار (X_test و y_test)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Model Mean Squared Error: {mse}")
