import cv2
import numpy as np
import glob
import os

# --- Configuration ---
INPUT_DIR = 'calibration/images2' # Aapki purani original images ka folder
OUTPUT_DIR = 'calibration/images2_enhanced'   # Naya folder jahan sharp images save hongi

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

image_files = glob.glob(f'{INPUT_DIR}/*.jpg')
if not image_files:
    print(f"Error: '{INPUT_DIR}' mein koi images nahi milin!")
    exit()

# Test ke liye pehli image load karein
test_img = cv2.imread(image_files[0])
# Screen par fit karne ke liye resize (sirf display ke liye)
h, w = test_img.shape[:2]
display_img = cv2.resize(test_img, (800, int(400 * (h/w))))

def update_image(val):
    pass

# Sliders wali window banayen
cv2.namedWindow('Enhancement Sliders')
cv2.createTrackbar('Contrast', 'Enhancement Sliders', 10, 30, update_image) # Default 1.0
cv2.createTrackbar('Sharpness', 'Enhancement Sliders', 0, 20, update_image) # Default 0

print("--- INSTRUCTIONS ---")
print("1. 'Contrast' aur 'Sharpness' sliders ko adjust karein.")
print("2. Jab black squares bilkul dark aur sharp nazar aayen, to 's' dabayen (Save All).")
print("3. Agar bina save kiye band karna ho to 'q' dabayen.")

# Interactive Loop
while True:
    c_val = cv2.getTrackbarPos('Contrast', 'Enhancement Sliders') / 10.0
    s_val = cv2.getTrackbarPos('Sharpness', 'Enhancement Sliders')

    # 1. Apply Contrast
    img_contrast = cv2.convertScaleAbs(display_img, alpha=c_val, beta=0)

    # 2. Apply Unsharp Masking (Professional Sharpening)
    if s_val > 0:
        blur = cv2.GaussianBlur(img_contrast, (0, 0), 3)
        img_final = cv2.addWeighted(img_contrast, 1.0 + (s_val*0.2), blur, -(s_val*0.2), 0)
    else:
        img_final = img_contrast

    cv2.imshow('Enhancement Sliders', img_final)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        final_c = c_val
        final_s = s_val
        break
    elif k == ord('q'):
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

print(f"\nValues Set -> Contrast: {final_c}, Sharpness: {final_s}")
print("Sari images ko process kiya ja raha hai, please wait...")

# Ab yahi formula aapki sari original bari images par apply hoga
for file in image_files:
    img = cv2.imread(file)
    
    # Apply Contrast
    enhanced = cv2.convertScaleAbs(img, alpha=final_c, beta=0)
    
    # Apply Sharpness
    if final_s > 0:
        blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
        enhanced = cv2.addWeighted(enhanced, 1.0 + (final_s*0.2), blur, -(final_s*0.2), 0)
    
    filename = os.path.basename(file)
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), enhanced)

print(f"Success! Sari improved images '{OUTPUT_DIR}' folder mein save ho gayin hain.")