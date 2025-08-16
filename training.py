from preprocess import load_clinical_data_with_patient_ids
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

CLINICAL_CSV2 = "clinical_data.csv"
df2 = pd.read_csv(CLINICAL_CSV2)
patient_ids2 = df2["PatientID"].tolist()

clinical_data, labels, feature_names = load_clinical_data_with_patient_ids(CLINICAL_CSV, patient_ids)

if clinical_data.shape[1] < 20:
    padding = np.zeros((clinical_data.shape[0], 20 - clinical_data.shape[1]))
    clinical_data = np.hstack([clinical_data, padding])

X_clinical_train, X_clinical_test, X_img_train, X_img_test, y_train, y_test = train_test_split(
    clinical_data, imaging_data, labels, test_size=0.2, random_state=42
)
input_layer = Input(shape=(clinical_data.shape[1],))
x = Dense(64, activation='relu')(input_layer)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_split=0.1)

os.makedirs("models/newModel", exist_ok=True)
model.save("models/newModel/asd_fusion_model2.h5")