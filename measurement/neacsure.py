import os
import cv2
import torch
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F
import glob
# --- CONFIGURATION ---
CALIB_FILE = 'calibration/camera_params.npz'
MODEL_WEIGHTS = 'model/simple_calculator_model.pth'
TEST_IMAGE = 'data/test2/*.jpg' # Pick an image from your dataset
TEST_IMAGE = 'data/test2/IMG_20260718_172207.jpg'
# ---------------------------------------------------------
# NEW REFERENCE OBJECT (1x2 inch Cardboard)
# 1 inch = 25.4 mm, 2 inches = 50.8 mm
# ---------------------------------------------------------
REF_HEIGHT_MM = 50.8
REF_WIDTH_MM = 25.4

def get_simple_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def main():
    print("Loading dependencies...")
    calib_data = np.load(CALIB_FILE)
    mtx, dist = calib_data['mtx'], calib_data['dist']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_simple_model(num_classes=2)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()
    # for img in glob.glob(TEST_IMAGE):
        # 1. READ & DYNAMICALLY RESIZE RAW IMAGE
    img_cv = cv2.imread(TEST_IMAGE)
    calib_w = int(mtx[0, 2] * 2) 
    calib_h = int(mtx[1, 2] * 2)
    img_cv = cv2.resize(img_cv, (calib_w, calib_h))

    # 2. UNDISTORT IMAGE
    undistorted_img = cv2.undistort(img_cv, mtx, dist, None, mtx)
    undistorted_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

    # 3. AI INFERENCE (Find Calculator)
    img_tensor = F.to_tensor(undistorted_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img_tensor)

    scores = prediction[0]['scores'].cpu().numpy()
    masks = prediction[0]['masks'].cpu().numpy()

    if len(scores) == 0 or scores[0] < 0.5:
        print("No calculator detected.")
        return

    confidence = scores[0] * 100
    best_mask = masks[0, 0] > 0.5
    mask_uint8 = (best_mask * 255).astype(np.uint8)
    
    calc_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    calc_contour = max(calc_contours, key=cv2.contourArea)
    calc_rect = cv2.minAreaRect(calc_contour)
    _, (calc_pixel_w, calc_pixel_h), calc_angle = calc_rect

    if calc_pixel_w > calc_pixel_h:
        calc_pixel_w, calc_pixel_h = calc_pixel_h, calc_pixel_w

    # 4. FIND THE REFERENCE CARDBOARD
    gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
    gray[mask_uint8 == 255] = 0  # Calculator ko black out kar dein
    
    # Threshold ko 180 se 210 kar diya taake floor ignore ho jaye aur sirf pure white card detect ho
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    
    ref_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for c in ref_contours:
        area = cv2.contourArea(c)
        # Filter 1: Area size (kacha kachra aur pure farsh ko ignore karein)
        if 500 < area < 50000:
            rect = cv2.minAreaRect(c)
            _, (cw, ch), _ = rect
            if cw > 0 and ch > 0:
                # Filter 2: Aspect Ratio (1x2 inch card ka ratio 2.0 hota hai)
                aspect_ratio = max(cw, ch) / min(cw, ch)
                if 1.5 < aspect_ratio < 2.5:
                    valid_contours.append(c)
    
    if not valid_contours:
        print("Error: Reference cardboard not found! Floor may be too bright or cardboard too small.")
        return
        
    # Jo contours filter pass kar lein, un mein se sab se behtar ko select karein
    card_contour = max(valid_contours, key=cv2.contourArea)
    card_rect = cv2.minAreaRect(card_contour)
    _, (card_pixel_w, card_pixel_h), _ = card_rect
    
    if card_pixel_w > card_pixel_h:
        card_pixel_w, card_pixel_h = card_pixel_h, card_pixel_w

    # 5. DERIVE RATIO FROM CARDBOARD
    pixels_per_mm = card_pixel_h / REF_HEIGHT_MM
    print(f"Ratio Computed: {pixels_per_mm:.2f} pixels per mm (from cardboard)")

    # 6. CALCULATE FINAL CALCULATOR MEASUREMENTS
    calculated_width_mm = calc_pixel_w / pixels_per_mm
    calculated_height_mm = calc_pixel_h / pixels_per_mm

    # 7. DRAW ANNOTATIONS FOR END-TO-END DEMO
    output_img = undistorted_img.copy()
    
    # Draw Blue Box for Reference Cardboard
    card_box = cv2.boxPoints(card_rect)
    card_box = np.int32(card_box)
    cv2.drawContours(output_img, [card_box], 0, (255, 0, 0), 2)
    cv2.putText(output_img, "REF: 50.8x25.4mm", (card_box[0][0]-50, card_box[0][1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Draw Red Mask & Green Box for Calculator
    colored_mask = np.zeros_like(output_img)
    colored_mask[:] = (0, 0, 255) 
    output_img[best_mask] = cv2.addWeighted(output_img[best_mask], 0.6, colored_mask[best_mask], 0.4, 0)

    calc_box = cv2.boxPoints(calc_rect)
    calc_box = np.int32(calc_box)
    cv2.drawContours(output_img, [calc_box], 0, (0, 255, 0), 2)

    # Add Metric Text Overlay
    cv2.putText(output_img, f"Conf: {confidence:.1f}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(output_img, f"W: {calculated_width_mm:.1f} mm", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(output_img, f"H: {calculated_height_mm:.1f} mm", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show Final Output
    output_resized = cv2.resize(output_img, (1000, 700))
    cv2.imshow("Final XIS Measurement Pipeline", output_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite(f"data/output/result_{os.path.basename(TEST_IMAGE)}.jpg", output_resized)
    
    print(f"--- MEASUREMENT RESULTS ---")
    print(f"Calculated Width: {calculated_width_mm:.2f} mm (Expected: ~152 mm)")
    print(f"Calculated Height: {calculated_height_mm:.2f} mm (Expected: ~198 mm)")

if __name__ == "__main__":
    main()