import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator


EYE_DATA_PATH = "dataset/processed_eyes"
MOUTH_DATA_PATH = "dataset/processed_mouth"

EYE_MODEL_PATH = "Models/eye_model.h5"
YAWN_MODEL_PATH = "Models/yawn_model.h5"

IMG_SIZE = 64
BATCH_SIZE = 16


eye_model = tf.keras.models.load_model(EYE_MODEL_PATH)
yawn_model = tf.keras.models.load_model(YAWN_MODEL_PATH)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

eye_val = datagen.flow_from_directory(
    EYE_DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

mouth_val = datagen.flow_from_directory(
    MOUTH_DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)


def evaluate(model, data, name):
    print("\n========================")
    print(f"Evaluating {name}")
    print("========================")

    y_true = data.classes
    y_pred_prob = model.predict(data)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(data.class_indices.keys())))

evaluate(eye_model, eye_val, "Eye Model")
evaluate(yawn_model, mouth_val, "Yawn Model")