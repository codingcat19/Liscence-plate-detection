import cv2
import pytesseract
import matplotlib.pyplot as plt

# Load the pre-trained Haar cascade for license plate detection
#pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"
plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

# Read the input image
img = cv2.imread(r"/Users/sahil/Code/Python/Liscence plate detection/mh1p.webp")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect license plates in the input image
plates = plate_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

states = {
    "Maharashtra": "MH", "Andhra Pradesh": "AP", "Arunachal Pradesh": "AR", "Assam": "AS",
    "Bihar": "BR", "Chattisgarh": "CG", "Delhi": "DL", "Goa": "GA", "Gujarat": "GJ",
    "Haryana": "HR", "Himachal Pradesh": "HP", "Jammu and Kashmir": "JK", "Jharkhand": "JH",
    "Karnataka": "KA", "Kerala": "KL", "Lakshadweep Islands": "LD", "Madhya Pradesh": "MP",
    "Manipur": "MN", "Meghalaya": "ML", "Mizoram": "MZ", "Nagaland": "NL", "Odisha": "OD/OR",
    "Pondicherry": "PY", "Punjab": "PB", "Rajasthan": "RJ", "Sikkim": "SK", "Tamil Nadu": "TN",
    "Telangana": "TS", "Tripura": "TR", "Uttar Pradesh": "UP", "Uttarakhand": "UK/UA",
    "West Bengal": "WB", "Andaman and Nicobar Islands": "AN", "Chandigarh": "CH",
    "Dadra & Nagar Haveli": "DN", "Daman & Diu": "DD", "Ladakh": "LA", "Other Territory": "OT"
}

# Process each detected license plate
for (x, y, w, h) in plates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Crop the license plate region
    plate_roi = gray_img[y:y + h, x:x + w]
    # Apply OCR to extract text from the license plate
    plate_number = pytesseract.image_to_string(plate_roi, config='--psm 7')
    print(f"Detected License Plate Number: {plate_number.strip()}")

    for state, code in states.items():
        if code in plate_number:
            detected_state = state
            detected_code = code
            break

plt.subplot(1, 1, 1)
plt.imshow(img, cmap='gray')
plt.title(f"Detected License Plate : {plate_number.strip()}")
plt.xlabel(f"State : {detected_state} ({detected_code})")

plt.show()
