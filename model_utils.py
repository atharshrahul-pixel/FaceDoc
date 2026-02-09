# model_utils.py
import io
import numpy as np
import cv2

# Load Haar cascade for face detection
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)


def _read_image_from_bytes(image_bytes):
    """Read image bytes into OpenCV BGR array safely."""
    try:
        image_bytes.seek(0)  # reset pointer
        file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] Could not read image bytes: {e}")
        return None


def _get_face_region(img):
    """Return bounding box (x, y, w, h) of largest detected face; fallback: whole image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        h, w = img.shape[:2]
        return (0, 0, w, h)
    # choose largest by area
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    return faces[0]


def _percent_mask_nonzero(mask):
    return 0 if mask.size == 0 else (np.count_nonzero(mask) / mask.size) * 100.0


def _detect_redness(face_bgr):
    """Estimate redness percentage in face region."""
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    # red has two hue ranges
    lower1 = np.array([0, 50, 30])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 50, 30])
    upper2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.medianBlur(red_mask, 5)
    red_percent = _percent_mask_nonzero(red_mask)
    return red_percent, red_mask


def _detect_pallor(face_bgr):
    """Approx pallor: low saturation + low R/G dominance."""
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))

    b, g, r = cv2.split(face_bgr.astype("float"))
    r_mean = float(np.mean(r))
    g_mean = float(np.mean(g))

    pallor_score = 0.0
    if s_mean < 45 and r_mean < (g_mean * 1.05):
        pallor_score = 100.0 - s_mean
    return pallor_score, {"s_mean": s_mean, "v_mean": v_mean, "r_mean": r_mean, "g_mean": g_mean}


def _detect_lesions(face_bgr):
    """Detect small dark spots (lesions)."""
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    th = cv2.adaptiveThreshold(
        l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 9
    )

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = l_channel.shape[:2]
    face_area = h * w
    lesion_area = 0
    lesion_count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if 10 < area < (face_area * 0.01):  # small lesions
            lesion_area += area
            lesion_count += 1

    lesion_percent = (lesion_area / face_area) * 100.0 if face_area else 0.0
    return lesion_percent, lesion_count


def _detect_eye_redness(face_bgr):
    """Estimate eye redness by inspecting upper third of face."""
    h, w = face_bgr.shape[:2]
    y1, y2 = int(h * 0.15), int(h * 0.45)
    eye_region = face_bgr[y1:y2, :]
    if eye_region.size == 0:
        return 0.0
    red_percent, _ = _detect_redness(eye_region)
    return red_percent


def analyze_image(image_bytes):
    """
    Analyze image bytes and return two-line string:
    Predicted label: <short label>
    Possible condition: <human-readable condition>
    """
    img = _read_image_from_bytes(image_bytes)
    if img is None:
        return "Predicted label: invalid_image\nPossible condition: Could not read image."

    # Detect face region
    x, y, w, h = _get_face_region(img)
    face = img[y:y + h, x:x + w]

    if face.size == 0:
        face = img

    # Resize for faster processing
    face_small = cv2.resize(face, (224, 224))

    # Heuristic analyses
    red_pct, _ = _detect_redness(face_small)
    pallor_score, pallor_meta = _detect_pallor(face_small)
    lesion_pct, lesion_count = _detect_lesions(face_small)
    eye_red_pct = _detect_eye_redness(face_small)

    # Decision logic
    condition = "Looks healthy!"
    label = "looks_healthy"
    confidence = 50.0

    if red_pct >= 12.0:
        label = "facial_redness"
        condition = "Inflammation / Allergy / Skin Redness"
        confidence = min(90.0, 50.0 + red_pct * 2.0)
    elif eye_red_pct >= 8.0:
        label = "eye_redness"
        condition = "Eye Strain / Inflamed Eye (possible conjunctivitis)"
        confidence = min(90.0, 50.0 + eye_red_pct * 2.0)
    elif lesion_pct >= 0.8 and lesion_count >= 3:
        label = "skin_lesions"
        condition = "Skin Issue / Lesions (acne, spots, pigmentation)"
        confidence = min(85.0, 50.0 + lesion_pct * 20.0)
    elif pallor_score > 40 and pallor_meta["v_mean"] > 150:
        label = "pale_appearance"
        condition = "Paleness (possible anemia or fatigue)"
        confidence = min(80.0, 45.0 + pallor_score * 0.5)
    elif red_pct > 4.0:
        label = "mild_redness"
        condition = "Mild Inflammation / Irritation"
        confidence = min(70.0, 50.0 + red_pct * 1.5)
    else:
        condition = f"Looks healthy! (Confidence: {round(100 - red_pct, 1)}%)"
        confidence = max(55.0, 100 - red_pct)

    label_text = (
        f"{label} (red={round(red_pct,2)}%, eye={round(eye_red_pct,2)}%, lesions={round(lesion_pct,2)}%)"
    )
    condition_text = f"{condition} (Confidence: {round(confidence,1)}%)"
    return f"Predicted label: {label_text}\nPossible condition: {condition_text}"
