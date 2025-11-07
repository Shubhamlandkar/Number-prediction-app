import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps, ImageDraw

# --- Train the Gradient Boosting Model ---
SEED = 23
X, y = load_digits(return_X_y=True)

train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=SEED)

gbc = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=100,
    max_features=5
)
gbc.fit(train_X, train_y)

# --- Streamlit UI ---
st.set_page_config(page_title="Digit Predictor", page_icon="🔢", layout="centered")
st.title("🧠 Handwritten Digit Prediction using Gradient Boosting")

st.write("Draw a digit (0–9) below and the model will predict it!")

# --- Create a drawing canvas ---
canvas_result = st.canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = Image.fromarray((img[:, :, 0]).astype(np.uint8))  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors: white digit on black background
    img = img.resize((8, 8))    # Resize to 8x8 like sklearn digits dataset
    img_array = np.array(img) / 16.0  # Normalize like original dataset
    img_array = img_array.flatten().reshape(1, -1)
    
    if st.button("🔍 Predict"):
        pred = gbc.predict(img_array)[0]
        st.success(f"✅ The model predicts this digit is: **{pred}**")

# --- Show model accuracy ---
pred_y = gbc.predict(test_X)
acc = accuracy_score(test_y, pred_y)
st.write(f"Model Accuracy on Test Data: **{acc:.2f}**")
