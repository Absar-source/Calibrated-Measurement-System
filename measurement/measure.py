import os
import cv2
import torch
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F
import math
import glob
# --- CONFIGURATION ---
CALIB_FILE = 'calibration/camera_params.npz'
MODEL_WEIGHTS = 'model/simple_calculator_model.pth'
TEST_IMAGE = 'data/test/*.jpg' # Pick an image from your dataset

# ---------------------------------------------------------
# CALIBRATION RATIO (Physical Ground Truth)
# You MUST measure your actual physical calculator with a ruler.
# Replace these values with your calculator's true physical dimensions.
# ---------------------------------------------------------
PHYSICAL_HEIGHT_MM = 198.0  # Example: 150mm tall
PHYSICAL_WIDTH_MM = 152.0   # Example: 120mm wide

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
    for img in glob.glob(TEST_IMAGE):
        # 1. READ & DYNAMICALLY RESIZE RAW IMAGE
        img_cv = cv2.imread(img)
        calib_w = int(mtx[0, 2] * 2) 
        calib_h = int(mtx[1, 2] * 2)
        img_cv = cv2.resize(img_cv, (calib_w, calib_h))

        # 2. UNDISTORT IMAGE (Mandatory XIS Requirement)
        undistorted_img = cv2.undistort(img_cv, mtx, dist, None, mtx)
        undistorted_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

        # 3. AI INFERENCE
        img_tensor = F.to_tensor(undistorted_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(img_tensor)

        scores = prediction[0]['scores'].cpu().numpy()
        masks = prediction[0]['masks'].cpu().numpy()

        if len(scores) == 0 or scores[0] < 0.5:
            print("No object detected with high confidence.")
            return

        confidence = scores[0] * 100
        print(f"Object Detected with {confidence:.1f}% confidence.")

        # 4. EXTRACT MASK & FIND ROTATED DIMENSIONS
        best_mask = masks[0, 0] > 0.5
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        
        # Find the exact pixel boundary of the mask
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the minimum area rotated rectangle
        rect = cv2.minAreaRect(largest_contour)
        (center_x, center_y), (pixel_width, pixel_height), angle = rect

        # Ensure width is the shorter side and height is the longer side
        if pixel_width > pixel_height:
            pixel_width, pixel_height = pixel_height, pixel_width

        # 5. DERIVE PIXEL-TO-MM RATIO
        # We calculate the ratio based on the known physical height
        pixels_per_mm = pixel_height / PHYSICAL_HEIGHT_MM
        
        # 6. CALCULATE FINAL METRIC MEASUREMENTS
        calculated_width_mm = pixel_width / pixels_per_mm
        calculated_height_mm = pixel_height / pixels_per_mm
        print("calculated_width_mm",calculated_width_mm)
        print("calculated_height_mm",calculated_height_mm)
        print("pixels_per_mm",pixels_per_mm)
        print("pixel_width",pixel_width)
        print("pixel_height",pixel_height)
        print("angle",angle)
        print("center_x",center_x)
        print("center_y",center_y)

        # 7. DRAW ANNOTATIONS FOR END-TO-END DEMO
        output_img = undistorted_img.copy()
        
        # Draw Red Mask Overlay
        colored_mask = np.zeros_like(output_img)
        colored_mask[:] = (0, 0, 255) 
        output_img[best_mask] = cv2.addWeighted(output_img[best_mask], 0.6, colored_mask[best_mask], 0.4, 0)

        # Draw Green Rotated Bounding Box
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(output_img, [box], 0, (0, 255, 0), 2)

        # Add Metric Text Overlay
        label_1 = f"Conf: {confidence:.1f}%"
        label_2 = f"W: {calculated_width_mm:.1f} mm"
        label_3 = f"H: {calculated_height_mm:.1f} mm"
        
        cv2.putText(output_img, label_1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(output_img, label_2, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(output_img, label_3, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show Final Output
        cv2.imshow("Metric Measurement Output", output_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        
        print(f"--- MEASUREMENT RESULTS ---")
        print(f"Calculated Width: {calculated_width_mm:.2f} mm")
        print(f"Calculated Height: {calculated_height_mm:.2f} mm")

if __name__ == "__main__":
    main()