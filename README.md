# Fashion-MNIST

Team Members
Hussah Almuzaini

Abdullah Aljubran

Majed Alsarawani

Jana Almalki

Description
This web application performs image clustering using PCA (Principal Component Analysis), K-Means, and DBSCAN algorithms. Users can upload images, and the system will:

Preprocess the image (convert to grayscale, resize, and flatten)

Apply PCA for dimensionality reduction

Cluster the image using both K-Means and DBSCAN algorithms

Display the clustering results in an elegant interface

Requirements
To run this application, you'll need:

Python 3.7+

Streamlit

OpenCV (cv2)

NumPy

scikit-learn

Joblib

Pillow (PIL)

Installation
Clone this repository

Install the required packages:

pip install streamlit opencv-python numpy scikit-learn joblib pillow
How to Run
Make sure you have all the required model files in the same directory:

pca_model.pkl

kmeans_model10.pkl

dbscan_model.pkl

Run the application using Streamlit:

streamlit run app.py
The application will open in your default web browser

Usage
Click on "Browse files" or drag and drop an image file (PNG, JPG, or JPEG)

The application will display your uploaded image in a styled frame

View the clustering results from both K-Means and DBSCAN algorithms

K-Means will always assign the image to a cluster

DBSCAN might classify the image as an outlier (cluster -1) if it doesn't fit well in any cluster

Features
Clean, responsive interface with custom styling

Handles image preprocessing automatically

Displays results in an easy-to-understand format

Works with various image formats
