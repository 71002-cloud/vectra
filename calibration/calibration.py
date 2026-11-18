import numpy as np
from pathlib import Path
import cv2

CHESSBOARD_SIZE = (6, 9)  # The length and without the outer layer
SQUARE_SIZE = 0.025  # The size of the square in meters
camera_index = [1, 2]

image_folder = (Path(__file__).resolve().parent / "calibration_images")
left_folder = image_folder / "left_camera"
right_folder = image_folder / "right_camera"

object_points = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)  # Makes a list of coordinates for every point set to 0 XYZ
object_points[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)  # Fills the x and y coordinates 

object_points *= SQUARE_SIZE  # Converts it to meters

left_images = []
right_images = []
object_points_list = []

left_files = sorted(left_folder.glob("*.png"))
right_files = sorted(right_folder.glob("*.png"))

image = cv2.imread(str(left_files[0]))
image_size = image.shape[1::-1]

right_lookup = {file.name: file for file in right_files}

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for left_file in left_files:
    right_file = right_lookup.get(left_file.name)

    if right_file is None:
        continue

    left_image = cv2.imread(str(left_file))
    right_image = cv2.imread(str(right_file))

    if left_image is None or right_image is None:
        continue

    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    left_success, left_corners = cv2.findChessboardCorners(
        left_gray, CHESSBOARD_SIZE
    )

    right_success, right_corners = cv2.findChessboardCorners(
        right_gray, CHESSBOARD_SIZE
    )

    if not left_success or not right_success:
        continue

    left_corners = cv2.cornerSubPix(left_gray, left_corners, (11, 11), (-1, -1), criteria)

    right_corners = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1), criteria)

    left_images.append(left_corners)
    right_images.append(right_corners)
    object_points_list.append(object_points.copy())

print(f"Valid image pairs: {len(left_images)}")

if not object_points_list:
    raise RuntimeError("No valid chessboard image pairs found")

left_error, left_matrix, left_distortion, left_rvecs, left_tvecs = cv2.calibrateCamera(object_points_list, left_images, image_size, None, None)
right_error, right_matrix, right_distortion, right_rvecs, right_tvecs = cv2.calibrateCamera(object_points_list, right_images, image_size, None, None)

stereo_error, left_matrix, left_distortion, right_matrix, right_distortion, R, T, E, F = cv2.stereoCalibrate(
    object_points_list,
    left_images,
    right_images,
    left_matrix,
    left_distortion,
    right_matrix,
    right_distortion,
    image_size,
    flags=cv2.CALIB_FIX_INTRINSIC)

np.savez(
    "stereo_calibration.npz",
    left_matrix=left_matrix,
    left_distortion=left_distortion,
    right_matrix=right_matrix,
    right_distortion=right_distortion,
    R=R,
    T=T)
