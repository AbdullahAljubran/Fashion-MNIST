import streamlit as st
import numpy as np
import cv2
import joblib
import base64

st.set_page_config(page_title="Image Clustering", layout="centered")

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
            margin: 20px auto;
            display: flex;
            justify-content: center;
            align-items: center;
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
    </style>
""", unsafe_allow_html=True)

pca_model = joblib.load("pca_model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")
dbscan_model = joblib.load("dbscan_model.pkl")

st.title("Image Clustering using PCA, KMeans, and DBSCAN")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    resized = cv2.resize(bw, (28, 28))
    flat = resized.flatten() / 255.0
    flat = flat.reshape(1, -1)

    pca_features = pca_model.transform(flat)
    kmeans_cluster = kmeans_model.predict(pca_features)[0]
    dbscan_cluster = dbscan_model.fit_predict(pca_features)[0]

    _, img_encoded = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(img_encoded).decode()

    st.markdown(f"""
        <div class="image-frame">
            <img src="data:image/png;base64,{img_base64}" />
        </div>
    """, unsafe_allow_html=True)

    dbscan_text = f"Cluster: {dbscan_cluster}" if dbscan_cluster != -1 else "Outlier (No Cluster)"
    st.markdown(f"""
        <div class="results-container">
            <div class="result-box">
                KMeans Result : <br>{kmeans_cluster}
            </div>
            <div class="result-box">
                DBSCAN Result : <br>{dbscan_text}
            </div>
        </div>
    """, unsafe_allow_html=True)
