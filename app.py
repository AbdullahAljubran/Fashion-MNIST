import streamlit as st
import numpy as np
import cv2
import joblib
import base64
import matplotlib.pyplot as plt
import io
from PIL import Image

st.set_page_config(page_title="Fashion MNIST Clustering", layout="centered")

st.markdown("""
    <style>
        html, body, .stApp {
            background-color: #fdf6ec;
            color: #4e3620;
        }

        .image-frame {
            width: 5cm;
            height: 5cm;
            padding: 8px;
            border: 4px solid #b99765;
            border-radius: 10px;
            background-color: #fff;
        }

        .image-frame img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 6px;
        }

        .results-container {
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
            gap: 20px;
        }

        .result-box {
            flex: 1;
            background-color: #e6d3b3;
            padding: 16px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            color: #4e3620;
            text-align: center;
        }

        .brown-button > button {
            background-color: #a97142;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

pca = joblib.load("pca_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")
cluster_map = joblib.load("cluster_map.pkl")

data = np.load("fashion_mnist_data.npz")
X = data['X']
y = data['y']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, -1), img

def predict_cluster(image_array):
    image_pca = pca.transform(image_array)
    cluster = kmeans.predict(image_pca)[0]
    predicted_label = cluster_map.get(cluster, -1)
    return cluster, predicted_label

def main():
    st.title("Fashion MNIST Clustering")

    uploaded_file = st.file_uploader("Upload a 28x28 grayscale image (png/jpg)", type=["png", "jpg", "jpeg"])

    if st.button("Show Random Example"):
        idx = np.random.randint(0, len(X))
        img = X[idx]
        flat_img = img.reshape(1, -1)
        cluster, pred_label = predict_cluster(flat_img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption=f"True Label: {class_names[y[idx]]}", width=150)
        with col2:
            st.markdown(f"""
                <div class="results-container">
                    <div class="result-box">
                        Cluster: {cluster}
                    </div>
                    <div class="result-box">
                        Predicted Category:<br> {class_names[pred_label] if pred_label != -1 else "Unknown"}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    if uploaded_file:
        processed_array, processed_img = preprocess_image(uploaded_file)
        cluster, pred_label = predict_cluster(processed_array)

        col1, col2 = st.columns(2)
        with col1:
            st.image(processed_img, caption="Uploaded Image", width=150)
        with col2:
            st.markdown(f"""
                <div class="results-container">
                    <div class="result-box">
                        Cluster: {cluster}
                    </div>
                    <div class="result-box">
                        Predicted Category:<br> {class_names[pred_label] if pred_label != -1 else "Unknown"}
                    </div>
                </div>
            """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
