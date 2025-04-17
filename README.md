# 🖼️ Image Recognition Using Deep Learning

## 🚀 Project Overview
This project implements an **Image Recognition Model** using **Convolutional Neural Networks (CNNs)** in **PyTorch**. The goal is to classify both **living and non-living objects** such as humans, animals, vehicles, and everyday objects.

We initially experimented with the **CIFAR-10 dataset** but have now moved on to a **custom real-world dataset** for improved accuracy and relevance.

---

## 📂 Dataset Information

### ✅ Custom Dataset
We are currently using a **custom dataset** located under:
```
real_dataset/
├── train/
│   ├── animals/
│   ├── humans/
│   └── vehicles/
├── test/
│   ├── animals/
│   ├── humans/
│   └── vehicles/
```

Each subfolder contains images classified by category. The dataset supports **class-wise balanced sampling** and **augmentation** techniques like random crop, flip, rotation, jitter, and affine transformations.

---

## 🧠 Model Architecture: `ImprovedCNN`
The model is defined in `train.py` and includes:

- ✅ 4 Convolutional Layers with Batch Normalization and ReLU
- ✅ MaxPooling after each convolutional block
- ✅ Dropout for regularization
- ✅ 3 Fully Connected Layers for classification
- ✅ Trained using `CrossEntropyLoss` and the `Adam` optimizer

---

## 🔍 Feature Map Visualization
You can now **visualize feature maps** from any convolutional layer (`conv1`, `conv2`, `conv3`, `conv4`) using the `utils.py` script.

### How to Use:
```bash
python src/utils.py
```
🔧 Set the correct image_path and model_path inside utils.py

📁 Output: Feature map visualizations will be saved as:
- feature_maps_conv1.png
- feature_maps_conv2.png
- feature_maps_conv3.png
- feature_maps_conv4.png

---

## 🧪 Evaluation & Insights
Training includes early stopping and learning rate scheduling

Model performance is evaluated using:
- 🔢 Class-wise accuracy
- 📉 Training & validation loss curves
- 🧩 Confusion matrix visualized as a heatmap

All metrics and plots are saved automatically for easy access

---

### Steps to Clone and Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/AkarshanGhosh/Image_recognisation.git
   cd Image_recognisation
   ```

2. Run the data_loader.py:
   ```bash
   python src/data_loader.py
   ```

3. Train the Model:
   ```bash
   python src/train.py
   ```

4. Visualize Feature Maps:
   ```bash
   python src/utils.py
   ```

5. Make Predictions:
   ```bash
   python src/predict.py 
   ```

---

## 📊 Results
- 🔥 High classification accuracy on custom dataset
- 🧠 Clear feature progression observed in feature maps from conv1 to conv4
- 📊 Clean visualizations of training loss and validation accuracy

---

## 👩‍💻 Contributors
- Akarshan Ghosh
  
---

## 📌 Future Work
- Add Grad-CAM based heatmap visualizations
- Create a Streamlit/Web UI for real-time predictions
- Extend dataset and support multi-label classification
