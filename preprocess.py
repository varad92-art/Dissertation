import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import SimpleITK as sitk

def load_clinical_data_with_patient_ids(filepath, patient_ids):
    df = pd.read_csv(filepath)
    df = df.set_index("PatientID").loc[patient_ids].reset_index()

    labels = df['ASD_Outcome'].values

    numeric_cols = ["Age", "BMI", "FusionSegments", "DiscAngle"]
    numeric = df[numeric_cols]

    categorical_cols = [
        "Gender", "Comorbidity_Hypertension", "Comorbidity_Diabetes",
        "SmokingStatus", "PhysicalActivity"
    ]
    categorical = df[categorical_cols].astype(str)

    # Impute and scale numeric features
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(imputer.fit_transform(numeric))

    # One-hot encode categoricals
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categorical = encoder.fit_transform(categorical)
    cat_feature_names = encoder.get_feature_names_out(categorical_cols)

    all_feature_names = list(numeric_cols) + list(cat_feature_names)
    features = np.concatenate([scaled_numeric, encoded_categorical], axis=1)

    return features, labels, all_feature_names

def load_clinical_data(filepath):
    # ... as before ...
    df = pd.read_csv(filepath)
    labels = df['ASD_Outcome'].values

    numeric = df[["Age", "BMI", "FusionSegments", "DiscAngle"]]
    categorical = df[["Gender", "Comorbidity_Hypertension", "Comorbidity_Diabetes"]].astype(str)

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(imputer.fit_transform(numeric))

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore',
        categories=[
            ['Male', 'Female', 'Other'], 
            ['Yes', 'No'], 
            ['Yes', 'No']
        ]
    )
    encoded_categorical = encoder.fit_transform(categorical)
    cat_feature_names = encoder.get_feature_names_out(["Gender", "Comorbidity_Hypertension", "Comorbidity_Diabetes"])
    all_feature_names = list(numeric.columns) + list(cat_feature_names)
    features = np.concatenate([scaled_numeric, encoded_categorical], axis=1)
    return features, labels, all_feature_names



def load_imaging_data(image_base_path, patient_ids, target_shape=(64, 64, 64), sequence="t1"):
    def resize_3d(img, target_shape):
        img_sitk = sitk.GetImageFromArray(img)
        orig_size = np.array(img.shape, dtype=np.int32)
        target_size = np.array(target_shape, dtype=np.int32)
        orig_spacing = img_sitk.GetSpacing()
        # Compute new spacing
        new_spacing = tuple(
            float(orig_spacing[i]) * float(orig_size[i]) / float(target_size[i])
            for i in range(3)
        )
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(tuple(map(int, target_shape)))
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetInterpolator(sitk.sitkLinear)
        img_resampled = resampler.Execute(img_sitk)
        return sitk.GetArrayFromImage(img_resampled)

    images = []
    for pid in patient_ids:
        # Extract numeric part from PatientID (e.g., PID_001 -> 1)
        try:
            pid_num = str(int(pid.split("_")[-1]))
        except Exception as e:
            print(f"[Error] Could not parse PatientID: {pid} ({e})")
            pid_num = str(pid)
        file_name = f"{pid_num}_{sequence}.mha"  # Use t1 by default
        file_path = os.path.join(image_base_path, file_name)
        if os.path.exists(file_path):
            img_sitk = sitk.ReadImage(file_path)
            img = sitk.GetArrayFromImage(img_sitk)
            img_resized = resize_3d(img, target_shape)
            img_resized = img_resized.astype(np.float32)
            img_resized = np.expand_dims(img_resized, axis=-1)
            images.append(img_resized)
        else:
            print(f"[Warning] Image not found for Patient ID: {pid} at {file_path}")
            images.append(np.zeros((*target_shape, 1), dtype=np.float32))
    return np.array(images)
