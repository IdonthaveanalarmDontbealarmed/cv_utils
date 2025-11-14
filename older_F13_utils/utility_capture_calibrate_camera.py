# Undistort fish-eye effect from the camera using OpenCV framework
# I strongly DO NOT recommend using it in the context of CV interference if this can be avoided
# by the means of training the model over an exactly-same or a very-similar camera footage.
# Distortions are fine. Your human eyes (and fish, and bird,..) have them for a reason.

# To undistort fish-eye effect from the camera:
#     undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs)
# To set-up propper coefficients: 
#     camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
#     dist_coeffs = np.array([k1, k2, p1, p2, k3])  
# Or load from file, created via this utility:
#     params = np.load('camera_params.npz
#     camera_matrix = params['camera_matrix']
#     dist_coeffs = params['dist_coeffs']

import cv2
import numpy as np
import glob

checkerboard_dims = (7, 5)  # Dimensions of the checkerboard (number of internal corners per row and column)

# Prepare object points (3D points in real world space)
objp = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Path to calibration images
images = glob.glob('cam_cal\\*.png')  
if not images: print("No images found. Check the path.")
else: print(f"Loaded {len(images)} images.")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, checkerboard_dims, corners, ret)
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(500)  # Wait for 500ms
cv2.destroyAllWindows()
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera Matrix:")
print(camera_matrix)
print("Distortion Coefficients:")
print(dist_coeffs)
np.savez('camera_params.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)