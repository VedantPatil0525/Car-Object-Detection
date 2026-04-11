# 🚗 Car Object Detection using CNN (ResNet50)

## 📌 Overview

This project implements a **Car Object Detection system** using **Convolutional Neural Networks (CNN)** with a pretrained **ResNet50** model.
The model predicts **bounding box coordinates (xmin, ymin, xmax, ymax)** for cars in images.

The project also compares the performance of different optimization algorithms:

* Adam
* SGD
* RMSprop

---

## 📂 Dataset

Dataset used: [Car Object Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/sshikamaru/car-object-detection)

### Dataset Structure

```
data/
├── training_images/          # Training images
├── testing_images/           # Images for prediction
└── train_solution_bounding_boxes.csv   # Bounding box labels
```

### CSV Format

```
image,xmin,ymin,xmax,ymax
img_1.jpg,50,30,200,180
```

---

## 🧠 Model Architecture

* Base Model: **ResNet50 (pretrained on ImageNet)**
* Custom Head:
  * Global Average Pooling
  * Dense Layers
  * Output Layer (4 neurons for bounding box regression)

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy, Pandas
* Matplotlib

---

## 🚀 Features

* Image preprocessing and bounding box scaling
* Transfer learning using ResNet50
* Training with multiple optimizers:
  * Adam
  * SGD
  * RMSprop
* Loss comparison visualization
* Bounding box prediction visualization
* Modular and clean code structure

---

## 📁 Project Structure

```
car-object-detection/
│
├── data/
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── visualize.py
│
├── models/        # Saved trained models
├── outputs/       # Graphs and prediction images
├── main.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the repository

```
git clone https://github.com/your-username/car-object-detection.git
cd car-object-detection
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the project

```
python main.py
```

---

## 📊 Results

### 🔹 Loss Comparison

* All optimizers converge over time
* Adam and RMSprop show faster convergence
* SGD initially unstable but improves later

### 🔹 Output Examples

* Loss graphs saved in `/outputs`
* Predicted bounding boxes drawn on images

---

## 📸 Sample Outputs

* Training loss graphs
* Predicted vs actual bounding boxes on images

---

## 📈 Evaluation Metric

* Mean Squared Error (MSE) for bounding box regression

---

## 💡 Future Improvements

* Add IoU (Intersection over Union) metric
* Use advanced models (YOLO, Faster R-CNN)
* Hyperparameter tuning
* Real-time detection using webcam
* Deploy as web app (Streamlit/Flask)

---

## 🙌 Acknowledgements

* Dataset from Kaggle
* TensorFlow & Keras documentation

---

## 👨‍💻 Author

Vedant Patil \
AI & DS Student

---

## ⭐ If you like this project

Give it a star on GitHub ⭐
