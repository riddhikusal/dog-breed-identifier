import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Use a verified Hugging Face model
MODEL_NAME = "wesleyacheng/dog-breeds-multiclass-image-classification-with-vit"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

st.title("🐶 Dog Breed Classifier")
st.write("Upload an image of a dog and I'll try to identify the breed!")

uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
        top_probs, top_idxs = torch.topk(probs, k=3)
        labels = [model.config.id2label[idx.item()] for idx in top_idxs]
        confidences = [float(p.item()) for p in top_probs]

    st.subheader("Top Predictions:")
    for label, conf in zip(labels, confidences):
        st.write(f"- **{label}** — {conf*100:.1f}% confidence")
