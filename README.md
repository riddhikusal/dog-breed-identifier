# 🐶 Dog Breed Classifier (POC)

This is a simple **Proof of Concept (POC)** project where a user can upload an image of a dog, and the app will predict the dog's breed using a **pretrained Vision Transformer (ViT) model** from [Hugging Face](https://huggingface.co).

---

## 🚀 Features
- Upload a dog image (`.jpg`, `.jpeg`, `.png`)
- Uses **`wesleyacheng/dog-breeds-multiclass-image-classification-with-vit`** model  
  - Trained on 120 dog breeds  
  - Provides top-3 predictions with confidence scores
- Simple **Streamlit web interface** that runs locally

---

## 🛠️ Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```



---

## 🔎 Code Explanation

### 1. **Imports**

```python
import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
```

* **Streamlit** → creates the simple web UI.
* **PIL (Pillow)** → handles image loading & conversion.
* **Torch** → used for tensor operations & running the deep learning model.
* **Transformers** → Hugging Face library for loading pretrained models.

---

### 2. **Load Pretrained Model**

```python
MODEL_NAME = "wesleyacheng/dog-breeds-multiclass-image-classification-with-vit"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
```

* **MODEL\_NAME** → Vision Transformer model fine-tuned on **120 dog breeds**.
* **AutoImageProcessor** → preprocesses images (resize, normalize, etc.) before sending to the model.
* **AutoModelForImageClassification** → loads the pretrained classification model.

---

### 3. **Streamlit UI Setup**

```python
st.title("🐶 Dog Breed Classifier")
st.write("Upload an image of a dog and I'll try to identify the breed!")
```

* Sets the app title and a description.
* This appears at the top of the web page.

---

### 4. **File Uploader**

```python
uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])
```

* Lets the user upload a file.
* Accepts only JPG/PNG formats.

---

### 5. **Handle Uploaded Image**

```python
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
```

* Opens the uploaded file using PIL.
* Converts to **RGB** (ensures consistent input format).
* Displays the image in the Streamlit app.

---

### 6. **Preprocess & Run Model**

```python
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
```

* **processor** → converts image into a PyTorch tensor suitable for the model.
* **with torch.no\_grad()** → disables gradient calculations (faster inference, less memory).
* **logits** → raw model outputs (one score per dog breed class).

---

### 7. **Convert Logits → Probabilities**

```python
probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
top_probs, top_idxs = torch.topk(probs, k=3)
```

* **softmax** → turns logits into probabilities (values between 0 and 1).
* **torch.topk** → picks the top-3 highest probabilities (best guesses).

---

### 8. **Map Predictions to Labels**

```python
labels = [model.config.id2label[idx.item()] for idx in top_idxs]
confidences = [float(p.item()) for p in top_probs]
```

* Each prediction index maps to a **dog breed name** (`id2label`).
* Confidence scores are extracted as floats.

---

### 9. **Display Results**

```python
st.subheader("Top Predictions:")
for label, conf in zip(labels, confidences):
    st.write(f"- **{label}** — {conf*100:.1f}% confidence")
```

* Displays the top 3 predictions with confidence percentages.
* Example output:

  ```
  Top Predictions:
  - Golden Retriever — 92.3% confidence
  - Labrador Retriever — 6.5% confidence
  - Flat-Coated Retriever — 1.2% confidence
  ```

---

