import streamlit as st
import numpy as np
import pandas as pd
# import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Activation, GRU, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
# from keras.utils import np_utils

# Load the dataset
data = pd.read_csv("./Dataset/Dementia_new_data.csv")

# Split into features and target
y = data['Dementia']
x = data.drop('Dementia', axis=1)

# Split into training and testing data
# X_train, X_test, y_train, y_test = train_test_split ( X, y,test_size = 0.2,random_state = 1,stratify = y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1,stratify = y)

# Standardize the data
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

# Build the model
# model = Sequential()
# model.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim=8))
# model.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))
# model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))


n_features, n_outputs = x_train_sc.shape[1], 2
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_features,1)))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(Dropout(0.2))
# dropout to prevent overfitting
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the trained model weights
model.load_weights("DementiaDetection_DL_Model_1DCNN_CSV.h5")

# Define class labels
class_labels = ['Non-Demented', 'Demented']

# Function to make predictions
# def predict_dementia(features):
#     # Preprocess the features
#     processed_features = sc.transform([features])

#     # Make predictions
#     prediction = model.predict(processed_features)[0]
#     predicted_index = np.argmax(prediction)

#     # Get the predicted class label
#     predicted_label = class_labels[predicted_index]

#     # Get the predicted class label
#     # predicted_label = class_labels[int(np.round(prediction))]

#     return predicted_label

def predict_dementia(features):
    # Preprocess the features
    processed_features = sc.transform([features])

    # Make predictions
    prediction = model.predict(processed_features)[0]

    # Get the predicted class index
    predicted_index = np.argmax(prediction)

    # Get the predicted class label
    predicted_label = class_labels[predicted_index]

    return predicted_label



# Streamlit app code
def main():
    st.image("./Logo.png", width=100)
    st.title("Dementia Classifier using MOD-1D-CNN")

    st.markdown("""
    <style>
        body {
            background-color: #f0f0f0; /* You can replace #f0f0f0 with any color code */
        }
    </style>
    """, unsafe_allow_html=True)
    # Set the app title and description
    # st.markdown("""<h1 style='text-align: center;'>Dementia Classifier using<br>MOD-1D-CNN</h1>""", unsafe_allow_html=True)
    # st.title("Dementia Classifier using MOD-1D-CNN")
    st.write("Enter the health metrics features and predict whether demented or non-demented.")
    st.write("**Health Metrics Input Format:**")
    bullet_points = [
    "**Diabetic status (0 for absence or 1 for presence)**",
    "**Age: 1(40-64) or 2(65-70) or 3(75-90)**",
    "**Heart rate: 1(<60bpm) or 2(60bpm – 100bpm) or 3(>100bpm)**",
    "**Blood oxygen level (1(<95%) or 2(95%-100%) or 3(>100%))**",
    "**Body temperature: 1(<36.5oC) or 2(36.5oC – 37.5oC) or 3(>37.5oC)**",
    "**Weight: 1(<50kg) or 2(50kg – 70kg) or 3(>70kg)**"
     ]

    st.write("- " + "\n- ".join(bullet_points))

    # Feature inputs
    Diabetic = st.radio("Diabetic", [0, 1])
    Age_Class = st.radio("Age_Class", [0, 1, 2])
    HeartRate_Class = st.radio("HeartRate_Class", [0, 1, 2])
    BloodOxygenLevel_Class = st.radio("BloodOxygenLevel_Class", [0, 1, 2])
    BodyTemperature_Class = st.radio("BodyTemperature_Class", [0, 1, 2])
    Weight_Class = st.radio("Weight_Class", [0, 1, 2])

    # age = st.number_input("Age", min_value=0)
    # educ = st.number_input("EDUC", min_value=0)
    # ses = st.number_input("SES", min_value=0)
    # mmse = st.number_input("MMSE", min_value=0)
    # cdr = st.number_input("CDR", min_value=0.0, max_value=1.0, step=0.1)
    # etiv = st.number_input("eTIV", min_value=0)
    # nwbv = st.number_input("nWBV", min_value=0.0, max_value=1.0, step=0.001)

    # Make predictions if all features are provided
    if st.button("Predict"):
        features = [Diabetic, Age_Class, HeartRate_Class, BloodOxygenLevel_Class, BodyTemperature_Class, Weight_Class]
        predicted_label = predict_dementia(features)
        st.write("Predicted Label:", predicted_label)

# Run the app
if __name__ == '__main__':
    main()
