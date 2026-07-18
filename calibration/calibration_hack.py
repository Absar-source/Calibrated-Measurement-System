import numpy as np
import cv2
import glob
import os

# 1. Puranay board ke inner corners ki exact tadad (5 columns, 8 rows)
CHECKERBOARD = (6, 8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = [] 

# 3D points generate karna
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# YAHAN APNI PURANI CALIBRATION IMAGES KA FOLDER PATH DEIN
images = glob.glob('calibration/images2 /*.jpg') 

print(f"Found {len(images)} images. Extracting corners...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Corners dhoondna
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                             cv2.CALIB_CB_FAST_CHECK + 
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

# Pehli kachi calibration (Iska error 1.5 ke qareeb aayega)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(f"\nInitial Error (with wrinkles): {ret:.4f}")

# ---------------------------------------------------------
# 2. THE FILTERING HACK: Ghalat images ko nikal bahar karein
# ---------------------------------------------------------
best_objpoints = []
best_imgpoints = []

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    
    # Agar kisi single image ka error 0.6 se zyada hai, to usay math se nikal dain!
    if error < 0.6:
        best_objpoints.append(objpoints[i])
        best_imgpoints.append(imgpoints[i])

# 3. Nayi Calibration sirf achi filtered images ke sath
if len(best_imgpoints) > 0:
    ret_final, mtx_final, dist_final, rvecs_final, tvecs_final = cv2.calibrateCamera(best_objpoints, best_imgpoints, gray.shape[::-1], None, None)
    
    print(f"Final Optimized Error: {ret_final:.4f} !!!")
    print(f"Used {len(best_imgpoints)} best images out of {len(objpoints)}.")
    
    # Save the clean parameters
    np.savez('camera_params.npz', mtx=mtx_final, dist=dist_final)
    print("Success: Clean matrix saved to 'camera_params.npz'.")
else:
    print("Error: Sab images bohat zyada kharab theen. Aapko margin thora barhana paray ga.")