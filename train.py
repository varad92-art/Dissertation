from preprocess import load_clinical_data_with_patient_ids, load_imaging_data
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, concatenate, Conv3D, MaxPooling3D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import os

CLINICAL_CSV = "clinical_data.csv"
IMAGE_DIR = "images/"
patient_ids = ["PID_001", "PID_002", "PID_003", "PID_004", "PID_005"]  # Extend as needed

clinical_data, labels, feature_names = load_clinical_data_with_patient_ids(CLINICAL_CSV, patient_ids)
imaging_data = load_imaging_data(IMAGE_DIR, patient_ids)

# Pad clinical data if < 20 features
if clinical_data.shape[1] < 20:
    padding = np.zeros((clinical_data.shape[0], 20 - clinical_data.shape[1]))
    clinical_data = np.hstack([clinical_data, padding])

X_clinical_train, X_clinical_test, X_img_train, X_img_test, y_train, y_test = train_test_split(
    clinical_data, imaging_data, labels, test_size=0.2, random_state=42
)

def clinical_branch(input_shape):
    x = Input(shape=input_shape)
    y = Dense(64, activation='relu')(x)
    y = Dropout(0.3)(y)
    y = Dense(32, activation='relu')(y)
    return x, y

def imaging_branch(input_shape):
    x = Input(shape=input_shape)
    y = Conv3D(16, (3, 3, 3), activation='relu')(x)
    y = MaxPooling3D(pool_size=(2, 2, 2))(y)
    y = Dropout(0.3)(y)
    y = Flatten()(y)
    y = Dense(64, activation='relu')(y)
    return x, y

clinical_input, clinical_out = clinical_branch((clinical_data.shape[1],))
image_input, image_out = imaging_branch((64, 64, 64, 1))
combined = concatenate([clinical_out, image_out])
z = Dense(32, activation='relu')(combined)
z = Dropout(0.3)(z)
z = Dense(1, activation='sigmoid')(z)

model = Model(inputs=[clinical_input, image_input], outputs=z)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit([X_clinical_train, X_img_train], y_train, epochs=10, batch_size=2, validation_split=0.1)

os.makedirs("models", exist_ok=True)
model.save("models/asd_fusion_model.h5")
