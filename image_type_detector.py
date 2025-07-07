import cv2
import numpy as np


def detect_image_type(img):
    """
    Detects whether the image is a screenshot, a photo of Screen, or a photo of paper receipt.

    Returns one of: 'Screenshot', 'phone_screen', 'paper_receipt'

    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 01 Detect glare or reflection

    glare_mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (0,0,220), (255, 255, 255))
    glare_pixels = cv2.countNonZero(glare_mask)
    glare_ratio = glare_pixels/ (h * w) 

    if glare_ratio > 0.02:
        return ' phone_screen'
    
    # 02 Detect paper like edges

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30 ,150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    if len(contours) >=3:
        biggest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(biggest, True)
        approx = cv2.approxPolyDP(biggest, 0.02 **peri, True)
        if len(approx) ==4:
            return'paper_reciept'
    
    # No_glare and no paper edges
    return 'Screenshot;'