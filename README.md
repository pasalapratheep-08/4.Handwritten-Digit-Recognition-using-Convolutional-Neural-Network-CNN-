# Create README.md using pypandoc as required
import pypandoc

text = r"""
# 🧠 Handwritten Digit Recognition using Convolutional Neural Network (CNN)

---

## 📌 1. Abstract

This project implements an image classification system capable of recognizing handwritten digits (0–9) using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset containing grayscale images of size 28×28 pixels. The goal is to demonstrate the complete machine learning pipeline including data exploration, preprocessing, model building, training, evaluation, visualization, and model saving.

---

## 📖 2. Introduction

Handwritten digit recognition is a fundamental computer vision problem widely used in postal mail sorting, bank cheque processing, and form digitization. The MNIST dataset is a standard benchmark dataset in deep learning consisting of handwritten digits from different individuals.

This project uses deep learning with Convolutional Neural Networks (CNNs), which automatically learn spatial features such as edges, curves, and shapes.

---

## 🗂️ 3. Dataset Description (MNIST)

* 📊 Total Images: 70,000
* 🏋️ Training Images: 60,000
* 🧪 Testing Images: 10,000
* 🖼️ Image Size: 28 × 28 pixels
* 🎨 Channels: Grayscale (1 channel)
* 🔢 Classes: 10 (Digits 0–9)

---

## 🔄 4. Project Workflow

1️⃣ Import Libraries
2️⃣ Load Dataset
3️⃣ Exploratory Data Analysis (EDA)
4️⃣ Data Preprocessing
5️⃣ Model Building (CNN)
6️⃣ Compile Model
7️⃣ Train Model
8️⃣ Evaluate Model
9️⃣ Visualization of Results
🔟 Save Trained Model

---

## 🔍 5. Exploratory Data Analysis

* Data shape and dimensions
* Class distribution
* Pixel intensity distribution
* Sample digit visualization

📌 Observations:

* Dataset balanced across all digits
* Pixel range: 0 – 255
* Writing style varies between samples

---

## ⚙️ 6. Data Preprocessing

### 🧼 Normalization

Pixel values converted from **[0,255] → [0,1]** for stable learning.

### 🔧 Reshaping

(28,28) → (28,28,1)

---

## 🏗️ 7. Model Architecture (CNN)

1. Conv2D (32 filters, 3×3, ReLU)
2. MaxPooling2D (2×2)
3. Dropout (0.25)
4. Conv2D (64 filters, 3×3, ReLU)
5. MaxPooling2D (2×2)
6. Dropout (0.25)
7. Flatten
8. Dense (128 neurons, ReLU)
9. Dropout (0.5)
10. Output Dense (10 neurons, Softmax)

🎯 CNN extracts edges, curves, and digit shapes automatically.

---

## 🧮 8. Model Compilation

* Optimizer: Adam ⚡
* Loss: Sparse Categorical Crossentropy 📉
* Metric: Accuracy 🎯

---

## 🏃 9. Model Training

* Epochs: 10
* Batch Size: 128
* Validation Split: 10%

Model learns patterns progressively while preventing overfitting.

---

## 📊 10. Model Evaluation

* Test Accuracy ✔️
* Confusion Matrix 🔢
* Precision / Recall / F1 Score 📏

Accuracy ≈ **99%**

---

## 📈 11. Visualizations

* Accuracy Curve 📉
* Loss Curve 📉
* Confusion Matrix 🔲
* Correct Predictions ✅
* Misclassified Images ❌

---

## 💾 12. Saving the Model

Saved as:
mnist_cnn_model.h5

Reusable without retraining.

---

## 🧰 13. Requirements

* numpy
* matplotlib
* seaborn
* tensorflow / keras
* sklearn

---

## ▶️ 14. How to Run

1. Open notebook
2. Run all cells
3. Model trains automatically
4. Results displayed
5. Model saved locally

---

## 🌍 15. Applications

* Bank cheque reading 🏦
* Postal code recognition 📮
* Form digitization 📝
* Automated number entry 🔢

---

## 🚀 16. Future Improvements

* Data Augmentation
* Deeper CNN
* Web App Deployment
* Mobile Model (TensorFlow Lite)

---

## 🏁 17. Conclusion

The CNN successfully learned handwritten digit patterns with high accuracy and demonstrates a complete deep learning workflow suitable for real-world applications.

---

**✨ End of Documentation ✨**
"""

output_path = "/mnt/data/README.md"
pypandoc.convert_text(text, 'md', format='md', outputfile=output_path, extra_args=['--standalone'])

output_path


