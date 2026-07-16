import numpy as np
import cv2
import glob
import os

# --- Configuration ---
# Your printed grid is 7x9 squares, which means 6x8 inner corners
CHECKERBOARD = (6, 8)
IMAGE_DIR = 'calibration/images/*.jpeg'  
OUTPUT_FILE = 'calibration/camera_params.npz'

def calibrate_camera():
    # Termination criteria for sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points: (0,0,0), (1,0,0), (2,0,0) ....,(6,8,0)
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all images
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane

    images = glob.glob(IMAGE_DIR)
    
    if not images:
        print("Error: No images found. Check your IMAGE_DIR path.")
        return

    print(f"Found {len(images)} images. Processing...")

    for fname in images:
        img = cv2.imread(fname)
        # img =  cv2.resize(img,(1080,720))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray_image, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp)
            # Refine corner locations
            corners2 = cv2.cornerSubPix(gray_image, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            # draw and display corners to verify 
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
            
    cv2.destroyAllWindows()

    print("Calculating calibration parameters...")
    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)

    # Save the intrinsic matrix and distortion coefficients
    np.savez(OUTPUT_FILE, mtx=mtx, dist=dist)
    
    print("\n--- Calibration Complete ---")
    print(f"Reprojection Error: {ret:.4f} pixels")
    print(f"Parameters saved to {OUTPUT_FILE}")
    
    if ret < 0.5:
        print("Status: EXCELLENT (Error is below 0.5)")
    elif ret < 1.0:
        print("Status: ACCEPTABLE (Error is below 1.0)")
    else:
        print("Status: POOR (Consider retaking photos with better lighting/flatness)")

if __name__ == "__main__":
    calibrate_camera()