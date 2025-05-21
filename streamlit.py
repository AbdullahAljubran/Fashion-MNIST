# app.py

import streamlit as st
import numpy as np
from PIL import Image
from joblib import load

# Load models and data
pca = load('pca_model.pkl')
kmeans = load('kmeans_model.pkl')
cluster_map = load('cluster_map.pkl')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load images and labels from saved .npz (must be same as used in pipeline)
data = np.load('fashion_mnist_data.npz')
X = data['X']  # shape: (num_samples, 28, 28)
y = data['y']

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, -1)

def predict_cluster(image_array):
    image_pca = pca.transform(image_array)
    cluster = kmeans.predict(image_pca)[0]
    predicted_label = cluster_map.get(cluster, -1)
    return cluster, predicted_label

def main():
    st.title("Fashion MNIST Clustering App")

    uploaded_file = st.file_uploader("Upload a 28x28 grayscale image (png/jpg)", type=["png", "jpg", "jpeg"])

    if st.button("Show Random Example"):
        idx = np.random.randint(0, len(X))
        img = X[idx]
        st.image(img, caption=f"True: {class_names[y[idx]]}", width=150)
        flat_img = img.reshape(1, -1)
        cluster, pred_label = predict_cluster(flat_img)
        st.write(f"Predicted Cluster: {cluster}")
        st.write(f"Likely Category: {class_names[pred_label]}")

    if uploaded_file:
        img_array = preprocess_image(uploaded_file)
        st.image(uploaded_file, caption="Uploaded Image", width=150)
        cluster, pred_label = predict_cluster(img_array)
        st.metric("Cluster", cluster)
        st.metric("Predicted Category", class_names[pred_label] if pred_label != -1 else "Unknown")

if __name__ == '__main__':
    main()
