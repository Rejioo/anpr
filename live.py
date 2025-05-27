import cv2
import pandas as pd
import easyocr
from ultralytics import YOLO
from datetime import datetime
import re
seen_plates=set()
PLATE_REGEX = re.compile(r'^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{3,4}$')
# === CONFIGURATION ===
MODEL_PATH = 'detect.pt'          # Your YOLOv8 model
CSV_PATH = 'new.csv'  # Output CSV
CONFIDENCE_THRESHOLD = 0.5        # Confidence threshold to filter weak detections

# === INIT ===
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'])
cap = cv2.VideoCapture(0)  # 0 = default camera

# Storage for OCR results
results_list = []

print("🔍 Starting camera... press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera feed error")
        break

    # === YOLO Detection ===
    detections = model(frame)[0]

    for box in detections.boxes:
        score = float(box.conf)
        if score < CONFIDENCE_THRESHOLD:
            continue  # Skip weak predictions

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = frame[y1:y2, x1:x2]

        # === OCR ===
        ocr_result = reader.readtext(cropped)
        print('reussss ',ocr_result)

        UNWANTED = {"IND", "INDIA", "BHARAT", "GOVT", "IN"}

        texts = [entry[1] for entry in ocr_result if entry[1].upper() not in UNWANTED]

        text = ''.join(texts).upper().replace(" ", "")

        # === Draw box & text ===
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
        text = text.upper().replace(" ", "")
        # === Store result ===
        # results_list.append({
        #     'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        #     'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        #     'Confidence': round(score, 2),
        #     'DetectedText': text
        # })
        if PLATE_REGEX.match(text):
            if text not in seen_plates:
                seen_plates.add(text)
                results_list.append({
                    'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'Confidence': round(score, 2),
                    'DetectedText': text
                })  
            print(f"[✅ VALID PLATE] {text}")
        else:
            print(f"[❌ REJECTED] {text} — doesn’t match plate format")

    # === Show frame ===
    cv2.imshow("Live Plate Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()

# === Save CSV ===
df = pd.DataFrame(results_list)
df.to_csv(CSV_PATH, index=False)
print(f"\n✅ Live session ended. {len(results_list)} plates logged to {CSV_PATH}")
