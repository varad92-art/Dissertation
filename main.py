from preprocess import load_clinical_data_with_patient_ids, load_imaging_data,load_clinical_data
from tensorflow.keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
from explainability import explain_with_shap, grad_cam, explain_with_lime

CLINICAL_CSV = "clinical_data.csv"
IMAGE_DIR = "images/"
MODEL_PATH = "models/asd_fusion_model.h5"
MODEL_PATH2="models/newModel/asd_fusion_model2.h5"

patient_ids = ["PID_001", "PID_002", "PID_003", "PID_004", "PID_005"]  # Extend as needed
features, labels, feature_names = load_clinical_data_with_patient_ids(CLINICAL_CSV, patient_ids)
X_images = load_imaging_data(IMAGE_DIR, patient_ids)

if features.shape[1] < 20:
    padding = np.zeros((features.shape[0], 20 - features.shape[1]))
    X_clinical_padded = np.hstack([features, padding])
    padded_feature_names = feature_names + [f"PAD_{i+1}" for i in range(20 - features.shape[1])]
else:
    X_clinical_padded = features
    padded_feature_names = feature_names

X_clinical, y, feature_names_1 = load_clinical_data(CLINICAL_CSV)
print("Shape of X_clinical:", X_clinical.shape)
if X_clinical.shape[1] < 20:
    padding = np.zeros((X_clinical.shape[0], 20 - X_clinical.shape[1]))
    X_clinical_padded_1 = np.hstack([X_clinical, padding])
    padded_feature_names_1 = feature_names_1 + [f"PAD_{i+1}" for i in range(20 - X_clinical.shape[1])]
else:
    X_clinical_padded_1 = X_clinical
    padded_feature_names_1 = feature_names_1

model = load_model(MODEL_PATH)
model2 = load_model(MODEL_PATH2)

# Create output directories
os.makedirs("gradcam_images", exist_ok=True)
os.makedirs("shap_images", exist_ok=True)
os.makedirs("lime_outputs", exist_ok=True)

dummy_image = np.zeros((64, 64, 64, 1))
dummy_clinical_vector = np.zeros((1, 20))


    
##explain_with_lime

clinical_width = X_clinical_padded.shape[1]  # e.g., 20
dummy_image2 = np.zeros((64,64,64,1), dtype="float32")  # or a representative volume
pids1=["PID_001", "PID_002"]
for idx, pid in enumerate(pids1):
    print(f"Processing {pid} ({idx+1}/{len(patient_ids)})")
    clinical_sample = X_clinical_padded[idx:idx+1]  # 1xN
    clean_html, png_path, probs = explain_with_lime(
        model=model2,
        X=clinical_sample,
        patient_id=pid,
        feature_names=padded_feature_names,
        dummy_image_input=dummy_image2,
        patient_image=X_images[idx],
        out_dir="lime_outputs",        
        class_names=('No ASD','ASD'),
        top_k=10
    )
    print(f"Saved: {clean_html}\n        {png_path}\n       probs={probs}")

## Explain with Shap
explain_with_shap(model, X_clinical_padded_1, default_image_input=dummy_image, sample_size=10, feature_names=padded_feature_names_1)
# shap_save_path = os.path.join("shap_images", f"shap_summary.pdf")
# plt.savefig(shap_save_path) 
# plt.close()


### Explain with gradcam
for idx, pid in enumerate(patient_ids):
    print(f"Processing {pid} ({idx+1}/{len(patient_ids)})")

    clinical_sample = X_clinical_padded[idx:idx+1]
    image_vol = X_images[idx]

    # ---- Grad-CAM ----
    gradcam_save_path = os.path.join("gradcam_images", f"gradcam_{pid}.png")
    grad_cam(
        model,
        image=image_vol,
        layer_name='conv3d',   # update to your layer name
        dummy_clinical_input=clinical_sample,   # patient-specific clinical data!
        alpha=0.4,
        patient_id=pid,
        save_path=gradcam_save_path
    )
    # print(f"Saved SHAP: {shap_save_path}, Grad-CAM: {gradcam_save_path}, LIME: {lime_save_path}")
