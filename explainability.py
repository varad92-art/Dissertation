import cv2
import shap
import lime
import lime.lime_tabular as ltb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
import os, base64, io




def explain_with_shap(model, X_clinical, default_image_input, sample_size=10, feature_names=None):
    # Pad feature_names to match columns
    n_features = X_clinical.shape[1]
    if feature_names is not None:
        if len(feature_names) < n_features:
            feature_names = feature_names + [f"PAD_{i+1}" for i in range(n_features - len(feature_names))]
        elif len(feature_names) > n_features:
            feature_names = feature_names[:n_features]

    # Convert to DataFrame
    if isinstance(X_clinical, np.ndarray):
        X_clinical_df = pd.DataFrame(X_clinical, columns=feature_names)
    else:
        X_clinical_df = X_clinical.copy()
        X_clinical_df.columns = feature_names

    sample = X_clinical_df.sample(n=min(sample_size, len(X_clinical_df)), random_state=42)

    def model_predict(clinical_input):
        batch_size = clinical_input.shape[0]
        image_input = np.repeat(np.expand_dims(default_image_input, axis=0), batch_size, axis=0)
        return model.predict([clinical_input, image_input]).flatten()

    explainer = shap.Explainer(model_predict, sample)
    shap_values = explainer(sample)

    # Filter out PAD columns for the plot
    real_features_mask = [not str(name).startswith("PAD_") for name in sample.columns]
    filtered_sample = sample.loc[:, real_features_mask]
    filtered_shap_values = shap_values[..., :sum(real_features_mask)]
    filtered_feature_names = list(filtered_sample.columns)

    print("Filtered feature names for SHAP plot:", filtered_feature_names)
    shap.summary_plot(filtered_shap_values, filtered_sample, feature_names=filtered_feature_names, plot_type="bar")


def grad_cam(model, image, layer_name, dummy_clinical_input, alpha=0.4, mask=None, patient_id=None, save_path=None):

    image_input = np.expand_dims(image, axis=0)
    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model([dummy_clinical_input, image_input])
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    conv_outputs = conv_outputs[0].numpy()
    grads = grads[0].numpy()

    weights = np.mean(grads, axis=(0, 1, 2))
    cam = np.zeros(conv_outputs.shape[:3], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[..., i]
    cam = np.maximum(cam, 0)
    cam_max = np.max(cam)
    if cam_max != 0:
        cam /= cam_max

    mid_slice = image.shape[0] // 2
    base_slice = image[mid_slice, :, :, 0]
    target_size = base_slice.shape[:2][::-1]
    heatmap = cv2.resize(cam[mid_slice], target_size)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    base_img = cv2.normalize(base_slice, None, 0, 255, cv2.NORM_MINMAX)
    base_img = np.uint8(base_img)
    base_img_rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_img_rgb, 1.0 - alpha, heatmap, alpha, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    title = "Grad-CAM Overlay on Center Slice"
    if patient_id: title += f"\\nPatient ID: {patient_id}"
    plt.title(title, fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    ax = plt.gca()
    mappable = plt.cm.ScalarMappable(cmap='jet')
    mappable.set_array(cam[mid_slice])
    plt.colorbar(mappable, ax=ax, fraction=0.02, pad=0.04, label="Model Attention")

    plt.show()
    if save_path:
        cv2.imwrite(save_path, overlay)

# def explain_with_lime(model, X, sample_size=1, dummy_image_input=None, feature_names=None):

#     if isinstance(X, np.ndarray):
#         X_df = pd.DataFrame(X, columns=feature_names)
#     else:
#         X_df = X.copy()

#     real_feature_mask = [not str(name).startswith("PAD_") for name in X_df.columns]
#     X_real = X_df.loc[:, real_feature_mask]
#     real_feature_names = [name for name in X_real.columns]

#     def predict_fn(clinical_array):
#         # clinical_array shape: (batch, n_real_features)
#         # You must pad this to (batch, 20) before passing to the model
#         batch = clinical_array.shape[0]
#         n_real = clinical_array.shape[1]
#         n_pad = 20 - n_real
#         if n_pad > 0:
#             clinical_array_padded = np.hstack([clinical_array, np.zeros((batch, n_pad))])
#         else:
#             clinical_array_padded = clinical_array
#         repeated_images = np.repeat(dummy_image_input[0:1], batch, axis=0)
#         preds = model.predict([clinical_array_padded, repeated_images])
#         preds = np.hstack((1 - preds, preds))  # for binary classification
#         return preds

#     explainer = lime.lime_tabular.LimeTabularExplainer(
#         training_data=X_real.values,
#         feature_names=real_feature_names,
#         class_names=['No ASD', 'ASD'],
#         mode='classification'
#     )

#     for i in range(sample_size):
#         exp = explainer.explain_instance(
#             X_real.iloc[i].values,
#             predict_fn,
#             num_features=min(10, X_real.shape[1])
#         )
#         exp.save_to_file(f'lime_explanation_{i}.html')


# def explain_with_lime(model,
#                       X,                              # 1xN (row) or small batch
#                       sample_size=1,
#                       dummy_image_input=None,         # e.g., (64,64,64,1)
#                       feature_names=None,
#                       out_dir="lime_outputs",
#                       patient_id=None,
#                       class_names=('No ASD','ASD'),
#                       top_k=10):
#     """
#     Generates TWO artifacts per patient:
#       1) Interactive HTML (LIME default)
#       2) Clean PNG bar chart with probabilities + top features

#     Works with a Keras multi-input model [clinical, image].
#     - Pads clinical vector to match the model's clinical input size.
#     - Repeats dummy image to match LIME's batches.

#     Returns: (html_path, png_path, probs)
#     """

#     os.makedirs(out_dir, exist_ok=True)

#     # --- Prepare clinical DataFrame (and hide PAD_* columns if present)
#     if isinstance(X, np.ndarray):
#         if feature_names is None:
#             feature_names = [f"feature_{i}" for i in range(X.shape[1])]
#         X_df = pd.DataFrame(X, columns=feature_names)
#     else:
#         X_df = X.copy()
#         if feature_names is None:
#             feature_names = list(X_df.columns)

#     mask_real = [not str(n).startswith("PAD_") for n in X_df.columns]
#     X_real = X_df.loc[:, mask_real]
#     real_names = list(X_real.columns)

#     # --- Model clinical input width (e.g., 20)
#     clinical_input_dim = int(model.inputs[0].shape[-1])  # assumes inputs=[clinical, image]
#     if clinical_input_dim is None:  # fallback if unknown
#         clinical_input_dim = X_df.shape[1]

#     # --- Build prediction wrapper for LIME
#     if dummy_image_input is None:
#         raise ValueError("dummy_image_input is required for multi-input model.")

#     # Ensure dummy_image_input is a single volume (no batch dim)
#     if dummy_image_input.ndim == 5:  # (B,D,H,W,C)
#         dummy_vol = dummy_image_input[0]
#     else:
#         dummy_vol = dummy_image_input

#     def predict_fn(clin_arr):
#         """Return 2-column probs for LIME: [P(No ASD), P(ASD)]."""
#         clin_arr = np.asarray(clin_arr, dtype=np.float32)
#         batch = clin_arr.shape[0]

#         # pad clinical to model width
#         n_real = clin_arr.shape[1]
#         n_pad = max(0, clinical_input_dim - n_real)
#         if n_pad > 0:
#             clin_arr = np.hstack([clin_arr, np.zeros((batch, n_pad), dtype=np.float32)])

#         # repeat dummy image for this batch
#         img_batch = np.repeat(dummy_vol[np.newaxis, ...], batch, axis=0)

#         # model returns (batch,1) -> convert to 2-column probs
#         p1 = model.predict([clin_arr, img_batch], verbose=0).reshape(-1, 1)
#         p = np.hstack([1.0 - p1, p1])  # [No ASD, ASD]
#         return p

#     # --- Build the LIME explainer on the available (real) features
#     explainer = lime.lime_tabular.LimeTabularExplainer(
#         training_data=X_real.values,
#         feature_names=real_names,
#         class_names=list(class_names),
#         mode='classification',
#         discretize_continuous=True,
#         kernel_width=3.0
#     )

#     # Explain the first (or only) row
#     row = X_real.iloc[0].values
#     exp = explainer.explain_instance(
#         row,
#         predict_fn,
#         num_features=min(top_k, X_real.shape[1]),
#         labels=[1]  # focus on ASD class
#     )

#     # --- Save interactive HTML
#     base = f"lime_explanation_{patient_id}" if patient_id is not None else "lime_explanation"
#     html_path = os.path.join(out_dir, f"{base}.html")
#     exp.save_to_file(html_path)

#     # --- Build a clean PNG bar chart
#     # predicted probabilities for this row
#     probs = predict_fn(row.reshape(1, -1))[0]  # [p_no_asd, p_asd]
#     pred_label = class_names[int(np.argmax(probs))]

#     # get weighted features for class 'ASD' (index 1)
#     contribs = exp.as_list(label=1)  # list of (feature_str, weight)
#     # turn into a DataFrame for sorting and plotting
#     dfw = pd.DataFrame(contribs, columns=['feature', 'weight'])
#     # Color by sign
#     colors = dfw['weight'].apply(lambda w: '#2ca02c' if w > 0 else '#d62728')  # green/red
#     # Order by absolute contribution
#     dfw = dfw.reindex(dfw['weight'].abs().sort_values(ascending=True).index)

#     plt.figure(figsize=(9, 6))
#     plt.barh(dfw['feature'], dfw['weight'], color=colors)
#     plt.axvline(0, color='k', linewidth=0.8)
#     plt.xlabel('Contribution to ASD (positive = pushes towards ASD)')
#     plt.title(f"LIME Explanation — Patient {patient_id if patient_id is not None else ''}\n"
#               f"P(No ASD)={probs[0]:.2f} | P(ASD)={probs[1]:.2f} | Predicted: {pred_label}")
#     plt.tight_layout()

#     png_path = os.path.join(out_dir, f"{base}.png")
#     plt.savefig(png_path, dpi=220)
#     plt.close()

#     return html_path, png_path, probs.tolist()
# explainability.py




def explain_with_lime(
    model,
    X,
    patient_id=None,
    feature_names=None,
    dummy_image_input=None,
    patient_image=None,
    out_dir="lime_outputs",
    class_names=('No ASD','ASD'),
    top_k=10,
    contrib_norm_threshold=1e-3,
    num_samples=300,           # LIME default is 5000, but for memory safety keep at 300-500
):
    """
    Saves TWO files per patient:
      1) <out_dir>/lime_explanation_<pid>.png
      2) <out_dir>/lime_explanation_<pid>_clean.html (embeds the PNG + table)
    Coloring rule:
      weight > 0  -> red  (#d62728)  pushes toward ASD
      weight < 0  -> green(#2ca02c)  pushes toward No ASD
    """

    os.makedirs(out_dir, exist_ok=True)
    # -------- Prepare clinical DataFrame --------
    if isinstance(X, np.ndarray):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
        if feature_names is None:
            feature_names = list(X_df.columns)
    # Hide PAD_* columns for the explainer
    real_mask = [not str(n).startswith("PAD_") for n in X_df.columns]
    X_real = X_df.loc[:, real_mask]
    real_names = list(X_real.columns)
    # Clinical input width expected by the model (assumes inputs=[clinical, image])
    clinical_input_dim = int(model.inputs[0].shape[-1]) if model.inputs[0].shape[-1] else X_df.shape[1]

    # Normalize dummy image to single volume (D,H,W,C)
    if dummy_image_input is None:
        raise ValueError("dummy_image_input is required for multi-input model.")
    dummy_vol = dummy_image_input[0] if dummy_image_input.ndim == 5 else dummy_image_input

    def predict_fn(clin_arr):
        clin_arr = np.asarray(clin_arr, dtype=np.float32)
        b = clin_arr.shape[0]
        n_real = clin_arr.shape[1]
        n_pad = max(0, clinical_input_dim - n_real)
        if n_pad:
            clin_arr = np.hstack([clin_arr, np.zeros((b, n_pad), dtype=np.float32)])
        p1 = model.predict(clin_arr, verbose=0).reshape(-1, 1)
        return np.hstack([1.0 - p1, p1])

    # -------- LIME explainer --------
    explainer = ltb.LimeTabularExplainer(
        training_data=X_real.values,
        feature_names=real_names,
        class_names=list(class_names),
        mode='classification',
        discretize_continuous=True,
        kernel_width=3.0
    )
    row = X_real.iloc[0].values
    exp = explainer.explain_instance(
        row,
        predict_fn,
        num_features=min(top_k, X_real.shape[1]),
        labels=[1],               # ASD class
        num_samples=num_samples
    )
    # Probabilities & predicted class
    probs = predict_fn(row.reshape(1, -1))[0]
    pred_label = class_names[int(np.argmax(probs))]
    pid_str = f"{patient_id}" if patient_id is not None else "sample"
    # -------- Parse LIME contributions for ASD class --------
    contribs = exp.as_list(label=1)
    dfw = pd.DataFrame(contribs, columns=["feat_str", "weight"])
    def _basename(s):
        for tok in ["<=", ">=", "<", ">", "="]:
            if tok in s:
                return s.split(tok)[0].strip()
        return s
    dfw["base_feat"] = dfw["feat_str"].apply(_basename)
    value_map = {n: X_real.iloc[0][n] for n in X_real.columns}
    dfw["value"] = dfw["base_feat"].map(value_map).astype(float)
    print(dfw[['base_feat','weight']])
    # Sort by |contribution|
    dfw = dfw.reindex(dfw["weight"].abs().sort_values(ascending=True).index)

    # -------- Plot (robust & readable) --------
    weights_true = dfw["weight"].astype(float).values
    y = np.arange(len(dfw))
    contrib_colors = ['#d62728' if w > 0 else '#2ca02c' for w in weights_true]  # red / green

    # Visual normalization for tiny contributions
    max_abs_true = float(np.nanmax(np.abs(weights_true))) if weights_true.size else 0.0
    VIS_NORM = max_abs_true < contrib_norm_threshold
    if VIS_NORM:
        w_plot = weights_true / (max_abs_true + 1e-12) if max_abs_true > 0 else weights_true
        xlim_contrib = 1.3
    else:
        w_plot = weights_true.copy()
        xlim_contrib = max_abs_true * 1.3 if max_abs_true > 0 else 1e-4

    v = dfw["value"].values
    v_scaled = v / (np.nanmax(np.abs(v)) + 1e-9) if np.nanmax(np.abs(v)) > 0 else v
    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    ax1.set_xlim(-xlim_contrib, xlim_contrib)
    ax1.set_xlabel("Contribution to ASD (positive → pushes towards ASD)")

    # Twin axis for value bars
    ax2 = ax1.twiny()
    ax2.set_zorder(1)
    ax1.set_zorder(2)
    ax2.patch.set_alpha(0.0)
    ax1.patch.set_alpha(0.0)
    ax2.barh(y, v_scaled, color="#bfe6bf", edgecolor="none", alpha=0.55, zorder=1)
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_xlabel("Feature value (scaled to [-1, 1])")

    # Contribution bars (foreground)
    ax1.barh(
        y, w_plot,
        color=contrib_colors, edgecolor="black", linewidth=0.9,
        alpha=0.98, zorder=3
    )
    ax1.axvline(0, color="k", lw=1.0, zorder=2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(dfw["feat_str"].tolist(), fontsize=10)
    # Annotate TRUE contribution and raw value
    for yi, w_true, w_disp, vs, raw in zip(y, weights_true, w_plot, v_scaled, v):
        xoff = 0.03 * (1 if w_disp >= 0 else -1) * (xlim_contrib if VIS_NORM else 1)
        ax1.text(w_disp + xoff, yi, f"{w_true:+.5g}",
                 va="center", ha="left" if w_disp >= 0 else "right",
                 fontsize=9, color="#222", zorder=4)
        ax2.text(vs + (0.03 if vs >= 0 else -0.03), yi, f"{raw:.2f}",
                 va="center", ha="left" if vs >= 0 else "right",
                 fontsize=9, color="#2f6f2f", zorder=4)
    # Legend & title
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#d62728", edgecolor="black", label="Contribution → ASD (+)"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="Contribution → No ASD (−)"),
        Patch(facecolor="#bfe6bf", edgecolor="none", label="Feature value (scaled)")
    ]
    ax1.legend(handles=legend_items, loc="lower right", frameon=True)
    title = (
        f"LIME Explanation — Patient {pid_str} | "
        f"P(No ASD)={probs[0]:.2f} | P(ASD)={probs[1]:.2f} | Predicted: {pred_label}"
    )
    if VIS_NORM:
        title += "  •  (contributions visually normalized)"
    plt.title(title, fontsize=12)
    plt.tight_layout()
    # Save PNG
    png_path = os.path.join(out_dir, f"lime_explanation_{pid_str}.png")
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    # -------- Clean HTML embedding PNG + small table --------
    def _img_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    img_b64 = _img_b64(png_path)
    rows_html = "\n".join(
        f"<tr><td>{n}</td><td style='text-align:right'>{val:.3f}</td>"
        f"<td style='text-align:right'>{w:+.5f}</td></tr>"
        for n, val, w in zip(dfw["base_feat"], dfw["value"], dfw["weight"])
    )
    clean_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>LIME — Patient {pid_str}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
h2 {{ margin: 0 0 10px 0; }}
table {{ border-collapse: collapse; width: 100%; max-width: 980px; }}
th, td {{ border-bottom: 1px solid #eee; padding: 6px 8px; }}
th {{ text-align: left; background: #fafafa; }}
.small {{ color: #555; }}
img {{ border:1px solid #ddd; max-width: 100%; height: auto; }}
</style>
</head>
<body>
  <h2>LIME — Patient {pid_str}</h2>
  <div class="small">
    P(No ASD) = {probs[0]:.3f} &nbsp;|&nbsp;
    P(ASD) = {probs[1]:.3f} &nbsp;|&nbsp;
    Predicted: <b>{pred_label}</b>
  </div>
  <div><img alt="LIME plot" src="data:image/png;base64,{img_b64}"/></div>
  <h3>Top features (this patient)</h3>
  <table>
    <thead>
      <tr><th>Feature</th><th style="text-align:right">Value</th><th style="text-align:right">Contribution to ASD</th></tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
  <p class="small">
    Colors: <b style="color:#d62728">red</b> bars push toward ASD;
    <b style="color:#2ca02c">green</b> bars push toward No ASD;
    light green bars show the feature value (scaled).
  </p>
</body>
</html>"""
    clean_html_path = os.path.join(out_dir, f"lime_explanation_{pid_str}_clean.html")
    with open(clean_html_path, "w", encoding="utf-8") as f:
        f.write(clean_html)
    return clean_html_path, png_path, probs.tolist()
