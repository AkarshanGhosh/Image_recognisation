# 🖼️ Image Recognition Using Deep Learning

## 🚀 Project Overview
This project implements an **Image Recognition Model** using **Convolutional Neural Networks (CNNs)** in **PyTorch**. The goal is to classify both **living and non-living objects** such as humans, animals, vehicles, and everyday objects.

We start with the **CIFAR-10 dataset** and will later train on a **custom dataset**.

---

## 📂 Dataset Information
We are using the **CIFAR-10 dataset**, which contains **60,000 color images** in **10 categories**:
- ✈️ Airplane  
- 🚗 Automobile  
- 🐦 Bird  
- 🐱 Cat  
- 🦌 Deer  
- 🐶 Dog  
- 🐸 Frog  
- 🐴 Horse  
- 🚢 Ship  
- 🚚 Truck  

The dataset is automatically downloaded when you run the `data_loader.py` script.

---



### Steps to Clone and Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/AkarshanGhosh/Image_recognisation.git
   cd Image_recognisation

2. Run the data_loader.py
   ```bash
   python src/data_loader.py

3. Train the Model
   ```bash
   python src/train.py

4. Make Predictions
   ```bash
   python src/predict.py --image <path_to_image>

