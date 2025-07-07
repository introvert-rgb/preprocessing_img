import cv2
import numpy as np

def detect_text_weight(img):
    """
    Analyze text density to classify as thin, regular, bold, or dot_matrix
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Step 1 : Binarize the image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #step 2 : Morph opening to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Step 3 Contour Analysis
    contours, _ =cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    density_list = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5 : #skip tiny noise
            continue
        roi = binary[y:y+h, x:x+w]
        area = w * h 
        text_pixels = cv2.countNonZero(roi)
        density = text_pixels / area
        density_list.append(density)
    
    if not density_list:
        return "unknown"
    
    avg_density = np.mean(density_list)

    # Step 4 Heuristics-based classification

    if avg_density < 0.18:
        return "dot_matrix"
    elif avg_density < 0.32:
        return "thin"
    elif avg_density < 0.50:
        return "regular"
    else:
        return "bold"
