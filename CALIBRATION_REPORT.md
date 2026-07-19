# Camera Calibration Report

## 1. Objective
The goal of this phase was to compute the intrinsic camera parameters and lens distortion coefficients to eliminate perspective and barrel/pincushion distortion, ensuring accurate 2D pixel-to-millimeter measurements.

## 2. Methodology
*   **Calibration Target:** A standard checkerboard pattern.
*   **Image Collection:** 20+ images were captured from multiple angles and distances to ensure robust parameter estimation.
*   **Algorithm:** Utilized OpenCV's `cv2.findChessboardCorners` for sub-pixel feature extraction and `cv2.calibrateCamera` to compute the camera matrix and distortion coefficients.

## 3. Results & Reprojection Error
The calibration was highly successful. The Root Mean Square (RMS) re-projection error is a key indicator of calibration accuracy.
*   **Initial Error (Unoptimized):** ~1.44 pixels
*   **Final Reprojection Error:** **0.31 pixels**

An error of `0.31` falls well within the "Excellent" category (typically < 0.5 pixels for industrial computer vision tasks), guaranteeing that the `cv2.undistort()` function will yield highly accurate rectilinear images for the measurement pipeline.

## 4. Artifacts
The resulting camera matrix (`mtx`) and distortion coefficients (`dist`) were saved as a serialized NumPy array (`camera_params.npz`) and are loaded dynamically during inference.