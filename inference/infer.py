import os
import cv2
import torch
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F
from PIL import Image

# --- CONFIGURATION PATHS ---
# Adjust these paths depending on where you run the script from your terminal
CALIB_FILE = 'calibration/camera_params.npz'
MODEL_WEIGHTS = 'model/simple_calculator_model.pth'
TEST_IMAGE = 'calibration/images/IMG_20260715_091609.jpeg' # Change this to your test image name

def get_simple_model(num_classes):
    """Rebuild the exact architecture used in training."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def main():
    print("1. Loading camera calibration parameters...")
    if not os.path.exists(CALIB_FILE):
        print(f"Error: Calibration file not found at {CALIB_FILE}")
        return
        
    calib_data = np.load(CALIB_FILE)
    mtx = calib_data['mtx']
    dist = calib_data['dist']

    print("2. Loading PyTorch model...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_simple_model(num_classes=2)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()

    print(f"3. Reading and undistorting test image: {TEST_IMAGE}")
    img_cv = cv2.imread(TEST_IMAGE)
    if img_cv is None:
        print("Error: Could not read test image.")
        return

    # MANDATORY STEP: Undistort the image before inference
    h, w = img_cv.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img_cv, mtx, dist, None, new_camera_mtx)
    
    # Crop the image based on the ROI to remove curved black edges
    x, y, w_roi, h_roi = roi
    undistorted_img = undistorted_img[y:y+h_roi, x:x+w_roi]

    print("4. Running AI Inference...")
    # Convert OpenCV image (BGR) to RGB PIL format for PyTorch
    img_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    img_tensor = F.to_tensor(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)

    scores = prediction[0]['scores'].cpu().numpy()
    
    if len(scores) == 0 or scores[0] < 0.7:
        print("No calculator detected with high confidence.")
        return

    print(f"Success! Calculator detected with {scores[0]:.2f} confidence.")
    
    # Extract best mask and bounding box
    masks = prediction[0]['masks'][0, 0].cpu().numpy()
    boxes = prediction[0]['boxes'][0].cpu().numpy().astype(int)

    # 5. Visualise the Output
    colored_mask = np.zeros_like(undistorted_img)
    colored_mask[:, :] = (0, 0, 255) # Red Mask
    
    binary_mask = masks > 0.5
    undistorted_img[binary_mask] = cv2.addWeighted(undistorted_img[binary_mask], 0.5, colored_mask[binary_mask], 0.5, 0)
    
    cv2.rectangle(undistorted_img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0), 2)
    cv2.putText(undistorted_img, f"Calculator: {scores[0]:.2f}", (boxes[0], boxes[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save and show
    output_path = 'inference_result.jpg'
    cv2.imwrite(output_path, undistorted_img)
    print(f"Saved annotated result to {output_path}")
    
    # Display the window (Press 'q' to close)
    # Resize for display purposes if the image is too large for your screen
    display_img = cv2.resize(undistorted_img, (int(w*0.5), int(h*0.5))) 
    cv2.imshow("Inference Output", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()