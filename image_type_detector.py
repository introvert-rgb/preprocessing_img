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
    hsv =cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #converts the img to Hue Saturation value format for detecting glare,bright spots and lighting issues
    glare_mask = cv2.inRange(hsv, (0,0,220), (255, 255, 255))
    glare_pixels = cv2.countNonZero(glare_mask) # countNonZero counts the pixels which are non-zero means which are not black from the array returned from inRange method of cv2
    glare_ratio = glare_pixels/ (h * w) 

    
    # 02 Detect paper like edges

    blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
    edged = cv2.Canny(blurred, 30 ,150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #the above code dealt  with detecting all the edges in image

    paper_found = False

    if len(contours) >=3:
        biggest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(biggest, True)
        approx = cv2.approxPolyDP(biggest, 0.02 **peri, True)  # this block helps in finding the biggest contour, then measuring its perimeter-closed shape, then if it then approxPolyDP reduces number of points in contour based on their proximity, if it has approx 4 points its a rectangle aka 'paper'.
        if len(approx) ==4:
            paper_found = True
    
    # No_glare and no paper edges check if its a screenshot
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var() 
    sharpness_threshold = 300 # screenshots are much sharper than photos
    #Laplacian operator, which detects edges in an image. we use 64 bit float so we don't loose precision, it gives img where edges are bright and flat/blurry regions are dark. var stands for variance there, sharp img will have high variance because of strong edge contrasts

    if paper_found:
        return 'paper_receipt'
    elif laplacian > sharpness_threshold:
        return "screenshots"
    elif glare_ratio > 0.02:
        return 'phone_screen'
    else:
        return 'Preferably Receipt'