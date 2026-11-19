import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras
from huggingface_hub import hf_hub_download

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

st.set_page_config(page_title="Brain MRI Classifier (Educational)")

@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="AbrishRizwan/brain-mri-resnet50",   
        filename="brain_mri_final.keras")
    return keras.models.load_model(path, compile=False)

st.title("Brain MRI Classifier (Educational Demo)")
st.warning("Educational project only. This is NOT a medical device and must NOT "
           "be used for diagnosis. Please do not upload real patient scans.")

with st.expander("Model limitations (please read)"):
    st.markdown("""
- Validation accuracy was 92%, but on a clean held-out test set (306 images, exact duplicates removed) accuracy was only **68%**.
- On that test set, glioma recall was **26%**; the model tends to predict meningioma.
- Grad-CAM showed the model often attends to skull borders or facial regions rather than the lesion.
- Classes: glioma, meningioma, pituitary tumor, no tumor.
""")

file = st.file_uploader("Upload a brain MRI image (JPG or PNG)",
                        type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB").resize((224, 224), Image.BILINEAR)
    st.image(img, caption="Uploaded image")
    x = np.expand_dims(np.array(img).astype("float32"), 0)
    probs = load_model().predict(x, verbose=0)[0]
    top = int(np.argmax(probs))
    st.subheader(f"Prediction: {CLASS_NAMES[top]} ({probs[top] * 100:.1f}% confidence)")
    st.bar_chart(pd.DataFrame({"probability": probs}, index=CLASS_NAMES))
    st.caption("A high confidence does not mean the prediction is correct.")
