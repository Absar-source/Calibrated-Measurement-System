# Dimensional Measurement & Error Analysis

## 1. The Measurement Pipeline
The final phase of the Calibrated Measurement System translates raw pixels into real-world metric dimensions (millimeters). 

**Step-by-Step Flow:**
1.  **Undistortion:** The raw image is passed through `cv2.undistort` using parameters from the calibration phase.
2.  **Target Segmentation (AI):** The Mask R-CNN model isolates the target object (e.g., calculator) and generates a precise pixel mask.
3.  **Reference Detection (Classical CV):** The AI mask is used to "black out" the target object, allowing classical OpenCV contour detection (`cv2.minAreaRect`, aspect ratio filtering) to reliably isolate the secondary reference object.
4.  **Metric Conversion:** A `pixels_per_mm` spatial ratio is derived from the reference object and applied to the target object.

## 2. The Hybrid Approach
To prevent "circular logic" (where an object acts as its own spatial reference), a calibrated secondary reference object was introduced: a **1x2 inch white cardboard** (25.4mm x 50.8mm).
*   **Why Hybrid?** Using deep learning for complex target segmentation and classical edge-detection/thresholding for the standardized reference object provides a lightweight, highly accurate, and robust pipeline without requiring a multi-class dataset retrain.

## 3. Error Analysis & Mean Absolute Error (MAE)
Because a rigid bird's-eye view (90-degree parallel to the floor) was utilized along with the undistortion matrix, perspective foreshortening was practically eliminated.

By using the independent cardboard reference, both the Width and Height of the target object dynamically adapt. 
*   **Expected Dimensions:** Width: ~152 mm, Height: ~198 mm
*   **Calculated Sample Output:** Width: [Insert a width from your test e.g., 141.4] mm, Height: [Insert a height from your test] mm.

*(Review the `inference_output` drive folder for visual proof of spatial calculations).*