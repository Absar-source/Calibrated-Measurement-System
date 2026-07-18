import os
import cv2
import torch
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIGURATION PATHS ---
CALIB_FILE = 'calibration/camera_params.npz'
MODEL_WEIGHTS = 'model/simple_calculator_model.pth'
TEST_IMAGE = 'data/IMG_20260715_134648.jpg' # Change this to your test image name

def get_simple_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def main():
    print("Loading parameters and model...")
    calib_data = np.load(CALIB_FILE)
    mtx, dist = calib_data['mtx'], calib_data['dist']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_simple_model(num_classes=2)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()

    # STEP 1: READ RAW IMAGE
    img_cv = cv2.imread(TEST_IMAGE)
    if img_cv is None:
        print(f"Error: Could not read {TEST_IMAGE}")
        return
    
    raw_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # STEP 2: UNDISTORT THE IMAGE
    h, w = img_cv.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img_cv, mtx, dist, None, new_camera_mtx)
    
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        undistorted_img = undistorted_img[y:y+h_roi, x:x+w_roi]
    
    undistorted_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

    # STEP 3: AI INFERENCE
    pil_img = Image.fromarray(undistorted_rgb)
    img_tensor = F.to_tensor(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)

    scores = prediction[0]['scores'].cpu().numpy()
    boxes = prediction[0]['boxes'].cpu().numpy()
    masks = prediction[0]['masks'].cpu().numpy()

    # Draw the AI predictions even if confidence is very low (0.1)
    output_rgb = undistorted_rgb.copy()
    guess_text = "No Detections"
    
    if len(scores) > 0:
        guess_text = f"Top Guess: {scores[0]*100:.1f}%"
        if scores[0] > 0.1:
            best_mask = masks[0, 0] > 0.5
            colored_mask = np.zeros_like(output_rgb)
            colored_mask[:, :] = (255, 0, 0) # Red overlay for matplotlib
            output_rgb[best_mask] = cv2.addWeighted(output_rgb[best_mask], 0.5, colored_mask[best_mask], 0.5, 0)
            
            box = boxes[0].astype(int)
            cv2.rectangle(output_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)

    # --- VISUAL DASHBOARD ---
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(raw_rgb)
    axes[0].set_title("Step 1: Raw Camera Image", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(undistorted_rgb)
    axes[1].set_title("Step 2: Undistorted Image\n(Does it look stretched?)", fontsize=14, color='yellow')
    axes[1].axis('off')

    axes[2].imshow(output_rgb)
    axes[2].set_title(f"Step 3: AI Output\n{guess_text}", fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    print("\nOpening Visual Dashboard...")
    plt.show()

if __name__ == "__main__":
    main()