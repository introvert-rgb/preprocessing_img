import cv2
from image_type_detector import detect_image_type
from text_weight_detector import detect_text_weight
from lighting_corrector import correct_lighting
from text_type_detector import detect_text_type
from preprocessing_selector import apply_best_preprocessing

# Load Image

image_path = r"F:\Github\preprocessing_img\test_images\TestPreReceipt011\TestPre011.jpg"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError("Image not found at path:", image_path)

# Stage 01 Image Type Detection

img_type = detect_image_type(img)
print(f"[Stage 1] Image Type: {img_type}")

# Stage 02 Text Weight Detection

weight = detect_text_weight(img)
print(f"[Stage 02] Text Weight : {weight}")

# # Stage 03 Lighting Correction

img_fixed = correct_lighting(img, img_type)

# # Stage 04 Text Type Detection

text_type = detect_text_type(img_fixed)
print(f"[Stage 04] Text Type: {text_type}")

# # Stage 05 Apply Smart Preprocessing

final_img = apply_best_preprocessing(img_fixed, img_type, weight, text_type)

# # Finally save the photo

cv2.imwrite('output/final_processed.jpg', final_img)
print("Preprocessing Done Image saved to output/final_processed.jpg")