import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
import LCD1602
from picamera2 import Picamera2
import time

# Config
frameWidth = 320
frameHeight = 240
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

# Setup camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (frameWidth, frameHeight)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# Load TFLite model
model = Interpreter(model_path="/home/pi/model.tflite")
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

# LCD setup
lcd = LCD1602.LCD1602(16, 2)
lcd.clear()

# Variabile pentru control afișare LCD
last_detected_time = 0
last_class_name = ""

# Preprocessing
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def equalize(img): 
    return cv2.equalizeHist(img)
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    img = img.astype(np.float32)
    return img

# Nume clase
def getClassName(classNo):
    classes = [
        'Viteza 20', 'Viteza 30', 'Viteza 50', 'Viteza 60', 'Viteza 70',
        'Viteza 80', 'Sf. viteza 80', 'Viteza 100', 'Viteza 120', 'Depasire interzisa',
        'Depasire >3.5t interzisa', 'Prioritate la intersectie', 'Drum prioritar', 'Cedeaza', 'STOP',
        'Acces interzis', 'Interzis >3.5t', 'Fara acces', 'Atentie', 'Curba stanga',
        'Curba dreapta', 'Curbe periculoase', 'Drum rau', 'Drum alunecos', 'Inghustare dreapta',
        'Lucrari drum', 'Semafor', 'Pietoni', 'Copii', 'Biciclete',
        'Gheata/zapada', 'Animale salbatice', 'Sf. restrictii', 'Dreapta inainte', 'Stanga inainte',
        'Inainte', 'Inainte/dreapta', 'Inainte/stanga', 'Tine dreapta', 'Tine stanga',
        'Sens giratoriu', 'Sf. depasire interzisa', 'Sf. depasire >3.5t'
    ]
    return classes[classNo]

# Detectare forma
def detect_shape(contour):
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    if len(approx) == 3: shape = "triangle"
    elif len(approx) == 4: shape = "rectangle"
    elif len(approx) > 4: shape = "circle"
    return shape

# LOOP principal
while True:
    imgOriginal = picam2.capture_array()
    imgOriginal = cv2.cvtColor(imgOriginal, cv2.COLOR_RGB2BGR)

    if imgOriginal is None:
        print("❌ Frame invalid!")
        continue

    imgROI = np.zeros((150, 150, 3), dtype=np.uint8)
    gray = grayscale(imgOriginal)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_in_this_frame = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500:
            shape = detect_shape(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if shape in ["circle", "triangle", "rectangle"]:
                roi = imgOriginal[y:y+h, x:x+w]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    roi_resized = cv2.resize(roi, (32, 32))
                    roi_processed = preprocessing(roi_resized)
                    roi_processed = roi_processed.reshape(1, 32, 32, 1)

                    model.set_tensor(input_details[0]['index'], roi_processed)
                    model.invoke()
                    predictions = model.get_tensor(output_details[0]['index'])

                    classIndex = np.argmax(predictions, axis=1)[0]
                    probabilityValue = np.amax(predictions)

                    if probabilityValue > threshold:
                        className = getClassName(classIndex)
                        print(f"✅ Detectat: {className} ({round(probabilityValue*100,2)}%)")

                        if className != last_class_name:
                            lcd.clear()
                            if len(className) <= 16:
                                lcd.setCursor(0, 0)
                                lcd.printout(className)
                            else:
                                lcd.setCursor(0, 0)
                                lcd.printout(className[:16])
                                lcd.setCursor(0, 1)
                                lcd.printout(className[16:32])
                            last_class_name = className


                        last_detected_time = time.time()
                        detected_in_this_frame = True

                        cv2.rectangle(imgOriginal, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(imgOriginal, f"{className} ({round(probabilityValue*100, 2)}%)",
                                    (x, y - 10), font, 0.6, (255, 0, 0), 2)
                        imgROI = cv2.resize(roi, (150, 150))

    # Dacă nu a detectat nimic și au trecut 3 secunde, șterge display-ul
    if not detected_in_this_frame and last_class_name != "":
        if time.time() - last_detected_time > 3:
            lcd.clear()
            last_class_name = ""

  #  cv2.imshow("Rezultat Final", imgOriginal)
  #  cv2.imshow("ROI", imgROI)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
lcd.clear()
picam2.stop()
#cv2.destroyAllWindows()
