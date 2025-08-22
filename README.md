# 🚦 Traffic Sign Recognition with Deep Learning

## 📌 Description
This project develops and evaluates deep learning models to classify German Traffic Signs using the GTSRB dataset.  
The work includes building a custom CNN from scratch and comparing it with a transfer learning model (MobileNetV2).  
The models are trained and validated on thousands of labeled traffic sign images to recognize 43 different categories.

---

## 📂 Dataset
The project uses the GTSRB - German Traffic Sign Recognition Benchmark dataset.

Includes:
- Training data → 31,367 labeled images across 43 classes (organized in folders).  
- Validation data → created via train/validation split.  
- Test data → 12,630 labeled images provided separately with a CSV.  

All images are real-world traffic sign photographs.

---

## 🔄 Preprocessing Steps
1. Load training images from class folders.  
2. Resize all images to 32×32.  
3. Normalize pixel values (scale 0–255 → 0–1).  
4. Convert labels to one-hot encoding.  
5. Split into train / validation / test sets.  
6. Apply data augmentation (rotation, shifting, shearing, zooming) to improve model generalization.  

---

## 🛠 Tools & Libraries
- Python  
- NumPy, Pandas → Data handling  
- OpenCV → Image preprocessing  
- Matplotlib → Visualization  
- scikit-learn → Data splitting & metrics  
- TensorFlow / Keras → Deep Learning (CNN & MobileNetV2)  

---

## 🧠 Model Architectures

### 1️⃣ Custom CNN
- Input: 32×32 RGB image  
- Conv2D + MaxPooling layers (32 → 64 → 128 filters)  
- Flatten → Dense(128) → Dropout → Dense(43, softmax)  

### 2️⃣ Transfer Learning (MobileNetV2)
- Pretrained MobileNetV2 (ImageNet weights, frozen base)  
- GlobalAveragePooling2D  
- Dense output layer with 43 softmax units  

---

## 📊 Training & Evaluation
- Optimizer: Adam  
- Loss function: Categorical Crossentropy  
- Metrics: Accuracy  

### Results:
| Model        | Validation Accuracy | Test Accuracy |
|--------------|----------------------|---------------|
| Custom CNN   | ~99.1 – 99.4 %       | 95.34 %   |
| MobileNetV2  | ~39.2 %              | 31.46 %   |

---

## 📈 Key Takeaways
- A simple CNN is highly effective for traffic sign recognition (95%+ test accuracy).  
- Data augmentation significantly improved generalization and validation accuracy.  
- MobileNetV2 performed poorly without fine-tuning (only 31% test accuracy).  
- The CNN model is suitable for real-world applications like driver-assistance systems.
