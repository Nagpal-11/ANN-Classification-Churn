import streamlit as st 
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder
import pandas as pd 
import pickle 

# loading the trained model 
model = tf.keras.models.load_model('model.h5')

with open('lable_encoder_gender.pkl' ,'rb') as file:
    lable_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl' ,'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl' ,'rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn Prediction')

## User inputs 
geography = st.selectbox('Geography' , onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender' , lable_encoder_gender.classes_)
age = st.slider('Age' , 18 , 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure' , 0 , 10)
num_of_products = st.slider('Number of Products' , 1 , 4)
has_cr_card = st.selectbox('Has Credit Card' , [0 ,1])
is_active_member = st.selectbox('Is Active Member' , [0,1])

# 1. Create the dictionary of inputs
input_dict = {
    'CreditScore': [credit_score],
    'Gender': [lable_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

# 2. Convert dictionary to a DataFrame first! (Fixed here)
input_df = pd.DataFrame(input_dict)

# One-hot encoded geography 
geo_encoded = onehot_encoder_geo.transform([[geography]])
# If your encoder is sparse, you might need to add .toarray() after transform()
if hasattr(geo_encoded, "toarray"):
    geo_encoded = geo_encoded.toarray()

geo_encoded_df = pd.DataFrame(geo_encoded , columns= onehot_encoder_geo.get_feature_names_out(['Geography']))

# 3. Safely combine them side-by-side
final_input_data = pd.concat([input_df.reset_index(drop = True) , geo_encoded_df] , axis = 1)

# Scale and predict
input_data_scaled = scaler.transform(final_input_data)
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

# Display results with a visual percentage divider
st.markdown("---")
if prediction_probability > 0.5:
    st.error(f'⚠️ The customer is likely to churn. (Risk Score: {prediction_probability:.1%})')
else:
    st.success(f'✅ The customer is not likely to churn. (Risk Score: {prediction_probability:.1%})')
