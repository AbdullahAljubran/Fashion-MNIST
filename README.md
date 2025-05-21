# Fashion-MNIST

## Team Members
- Hussah Almuzaini  
- Abdullah Aljubran  
- Majed Alsarawani  
- Jana Almalki  

## 📌 Description
This web application performs image clustering on Fashion-MNIST-like data using **Principal Component Analysis (PCA)**, **K-Means**, and **DBSCAN** algorithms. Users can upload an image, and the system will:

- ✅ Preprocess the image (convert to grayscale, resize, and flatten)  
- ✅ Apply PCA for dimensionality reduction  
- ✅ Perform clustering using both K-Means and DBSCAN  
- ✅ Display the results in a clean, responsive web interface  

---

## ⚙️ Requirements

To run this application, you'll need:

- Python 3.7+
- Streamlit  
- OpenCV (`opencv-python`)  
- NumPy  
- scikit-learn  
- Joblib  
- Pillow (`PIL`)  

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

   Install the required packages:

bash
Copy
Edit
pip install streamlit opencv-python numpy scikit-learn joblib pillow
▶️ How to Run
Make sure you have the following model files in the same directory as app.py:

pca_model.pkl

kmeans_model10.pkl

dbscan_model.pkl

Then, run the app using Streamlit:

bash
Copy
Edit
streamlit run app.py
The app will automatically open in your default web browser.

🚀 Usage
Click on "Browse files" or drag and drop an image (PNG, JPG, JPEG).

The uploaded image will be displayed in a styled frame.

View clustering results from both K-Means and DBSCAN:

K-Means will always assign a cluster label.

DBSCAN might classify the image as an outlier (cluster -1).

✨ Features
🖼️ Clean, responsive interface with custom styling

🧠 Automatic image preprocessing

📊 Clustering results displayed in a user-friendly format

🌐 Supports multiple image formats


