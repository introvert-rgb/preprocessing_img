import cv2
from image_type_detector import detect_image_type
from text_weight_detector import detect_text_weight

image_path = r"F:\Github\preprocessing_img\test_images\TestPreReceipt011\TestPre011.jpg"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError("Image not found at path:", image_path)

# Stage 01 Image Type Detection

img_type = detect_image_type(img)
print(f"[Stage 1] Image Type: {img_type}")

weight = detect_text_weight(img)
print(f"[Stage 02] Text Weight : {weight}")