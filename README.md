# 😴 Drowsiness Detection System

A deep learning-based system that detects drowsiness using **eye state** and **yawning detection**.

This project uses **Computer Vision + CNN models** to identify signs of fatigue and can be extended for real-time driver monitoring systems.

---

## 🚀 Overview

Drowsiness is detected using two key indicators:

- 👁 **Eye State Detection** → Open / Closed
- 😮 **Yawn Detection** → Yawn / No Yawn

Two separate CNN models are trained and used together for better accuracy.

---

## 🧠 Models Used

### 1. Eye Detection Model
- Input: 64x64 eye images  
- Classes:
  - Open  
  - Closed  
- Output: Binary classification  

### 2. Yawn Detection Model
- Input: 64x64 mouth images  
- Classes:
  - Yawn  
  - No Yawn  
- Output: Binary classification  

---

## 📂 Project Structure
Drowsiness-Detection/
│
├── dataset/
│ ├── processed_eyes/
│ │ ├── Open/
│ │ └── Closed/
│ │
│ ├── processed_mouth/
│ │ ├── yawn/
│ │ └── no_yawn/
│ │
│ └── preprocess_dataset.py
│
├── Models/
│ ├── eye_model.h5
│ └── yawn_model.h5
│
├── detection.py
├── train_eye_model.py
├── train_yawn_model.py
├── evaluate_model.py
├── .gitignore
└── README.md


---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Ayushii-Kumari/Drowsiness-Detection.git
cd Drowsiness-Detection
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
- pip install -r requirements.txt

 ### 4. Training the Models
- Train Eye Model
  ``` bash 
  python train_eye_model.py
  ```
  
- Train Yawn Model
 ``` bash 
  python train_yawn_model.py
  ```

### 5. Models will be saved in:
- Models/
  
### 6. Model Evaluation
``` bash 
python evaluate_model.py
```

This will output:
- Confusion Matrix
- Classification Report
  
### 7. Detection
``` bash 
python detection.py
```

Detects:
- Eye closure
- Yawning

🧪 Dataset

The dataset contains:
- Eye images → Open / Closed
- Mouth images → Yawn / No Yawn

⚠️ Dataset is not included due to size limitations.
You can use your own dataset or download from external sources.

🧠 Model Architecture
- Convolutional Neural Network (CNN)
- Layers used:
  i) Conv2D
  ii) MaxPooling
  iii) Dense
  iv) Dropout
- Optimizer: Adam
- Loss Function: Binary Crossentropy

📌 Key Features
- Separate models for eye and yawn detection
- Data augmentation for better generalization
- Validation split for evaluation
- Early stopping to prevent overfitting

🔮 Future Improvements
- Real-time webcam integration
- Alarm/alert system
- Mobile or web deployment
- Improved accuracy with larger dataset
