import os
import cv2
import numpy as np


BASE_PATH = "dataset"

OPEN_PATH = os.path.join(BASE_PATH, "Open")
CLOSED_PATH = os.path.join(BASE_PATH, "Closed")
YAWN_PATH = os.path.join(BASE_PATH, "yawn")
NO_YAWN_PATH = os.path.join(BASE_PATH, "no_yawn")

OUT_EYE_OPEN = os.path.join(BASE_PATH, "processed_eyes", "Open")
OUT_EYE_CLOSED = os.path.join(BASE_PATH, "processed_eyes", "Closed")
OUT_MOUTH_YAWN = os.path.join(BASE_PATH, "processed_mouth", "yawn")
OUT_MOUTH_NO_YAWN = os.path.join(BASE_PATH, "processed_mouth", "no_yawn")

for path in [OUT_EYE_OPEN, OUT_EYE_CLOSED, OUT_MOUTH_YAWN, OUT_MOUTH_NO_YAWN]:
    os.makedirs(path, exist_ok=True)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def safe_read_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return None

    # If grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # If BGRA / 4 channel
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # If weird dtype
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    return img


def preprocess_image(img, size=(64, 64)):
    if img is None or img.size == 0:
        return None

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, size)
        return img
    except Exception:
        return None


def process_folder(input_folder, output_folder, prefix):
    saved = 0
    skipped = 0

    if not os.path.exists(input_folder):
        print(f"Folder not found: {input_folder}")
        return

    for file_name in os.listdir(input_folder):
        if not file_name.lower().endswith(VALID_EXTENSIONS):
            continue

        img_path = os.path.join(input_folder, file_name)
        img = safe_read_image(img_path)

        if img is None:
            skipped += 1
            print(f"Skipping unreadable: {img_path}")
            continue

        processed = preprocess_image(img)

        if processed is None:
            skipped += 1
            print(f"Skipping bad image: {img_path}")
            continue

        out_name = f"{prefix}_{saved}.jpg"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, processed)
        saved += 1

    print(f"{prefix} done -> saved: {saved}, skipped: {skipped}")


if __name__ == "__main__":
    print("Processing eyes...")
    process_folder(OPEN_PATH, OUT_EYE_OPEN, "open")
    process_folder(CLOSED_PATH, OUT_EYE_CLOSED, "closed")

    print("\nProcessing mouth...")
    process_folder(YAWN_PATH, OUT_MOUTH_YAWN, "yawn")
    process_folder(NO_YAWN_PATH, OUT_MOUTH_NO_YAWN, "no_yawn")

    print("\n✅ Preprocessing complete.")