# Hidden_image_detector
📌 Project Overview

This project uses Convolutional Neural Network (CNN) to detect hidden images inside paintings.
The dataset contains oil painting images where:
Some images contain hidden blended images
Some images are normal oil paintings
The goal is to classify paintings as:
Hidden Image Present
No Hidden Image

📂 Dataset Structure

Dataset folder is organized into three parts:

dataset/
│
├── train/
│   ├── hidden/
│   └── no_hidden/
│
├── test/
│   ├── hidden/
│   └── no_hidden/
│
└── val/
    ├── hidden/
    └── no_hidden/
