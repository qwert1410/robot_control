import os
import glob
import numpy as np
import cv2
from cv2 import aruco


def apply_projective_transform(image, H, output_size, visualize=False):
    """
    Apply a projective transformation to the image using a homography matrix H.
    """
    H_inv = np.linalg.inv(H)
    transformed_image = cv2.warpPerspective(image, H, output_size)

    if visualize:
        image_small = cv2.resize(image, (0, 0), None, 0.5, 0.5)
        transformed_image_small = cv2.resize(transformed_image, (0, 0), None, 0.5, 0.5)
        imstack = np.hstack([image_small, transformed_image_small])
        cv2.imshow('Transformation Result', imstack)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    return transformed_image


def compute_homography(points_src, points_dst):
    """
    Compute the homography matrix that maps points_src to points_dst.
    """
    assert points_src.shape[0] == points_dst.shape[0], "Mismatch in source and destination points."
    assert points_src.shape[0] >= 4, "At least 4 points are required."

    A = []
    for (x, y), (u, v) in zip(points_src, points_dst):
        A.extend([
            [-x, -y, -1, 0, 0, 0, u * x, u * y, u],
            [0, 0, 0, -x, -y, -1, v * x, v * y, v],
        ])

    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape(3, 3)
    return H / H[2, 2]


def test_homography():
    """
    Test homography computation with random points.
    """
    for _ in range(10):
        H_true = np.random.rand(3, 3)
        H_true[2, 2] = 1

        num_points = 8
        points_src = np.random.rand(num_points, 2) * 100
        points_src_homogeneous = np.hstack([points_src, np.ones((num_points, 1))])
        points_dst_homogeneous = (H_true @ points_src_homogeneous.T).T
        points_dst = points_dst_homogeneous[:, :2] / points_dst_homogeneous[:, [2]]

        H_estimated = compute_homography(points_src, points_dst)
        H_true /= H_true[2, 2]
        H_estimated /= H_estimated[2, 2]

        assert np.allclose(H_true, H_estimated, atol=1e-6), "Homography estimation failed!"

    print("All tests passed!")


def get_homography(img_1, img_2, conf=0.95):
    """
    Retrieve the homography matrix between two images.
    """
    matches = np.load(f'matches/img{img_1}_img{img_2}_matches.npz')
    points_src = matches['keypoints0']
    points_dst = matches['keypoints1']
    mapping = matches['matches']
    confidence = matches['match_confidence']

    valid_indices = (mapping != -1) & (confidence > conf)
    filtered_points_src = points_src[valid_indices]
    filtered_points_dst = points_dst[mapping[valid_indices]]

    return compute_homography(filtered_points_src, filtered_points_dst)


def get_panorama(img_1, img_2, H, add_img_1=True):
    """
    Stitch two images using a homography matrix.
    """
    dst = apply_projective_transform(img_2, H, (img_1.shape[1] + img_2.shape[1], img_1.shape[0]))
    if add_img_1:
        dst[0:img_1.shape[0], 0:img_1.shape[1]] = img_1
    return dst


def crop_img(img):
    """
    Crop the black borders from an image.
    """
    max_w = np.max(np.where(img != 0)[1])
    return img[:, :max_w, :]


def col_diff(a, b):
    """
    Compute color difference between two pixels.
    """
    diff = np.abs(a - b)
    return np.sum(np.square([0.3, 0.59, 0.11] * diff))


def calibrate_camera(images, aruco_board, mapping):
    """
    Calibrate the camera using ArUco markers.
    """
    objpoints, imgpoints = [], []

    for image_path in images:
        image = cv2.imread(image_path)
        corners, ids, _ = detector.detectMarkers(image)

        if ids is not None:
            ids = ids.flatten()
            ids = [mapping.get(id, -1) for id in ids]

            ordered_corners = [None] * len(corners)
            for i, pos in enumerate(ids):
                if pos != -1:
                    ordered_corners[pos] = corners[i]

            points = np.vstack([corner for corner in ordered_corners if corner is not None])
            imgpoints.append(points.astype('float32'))
            objpoints.append(aruco_board)

    return cv2.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)


def undistort_images(images, mtx, dist, output_folder):
    """
    Undistort images using the camera calibration matrix.
    """
    for img_path in images:
        img = cv2.imread(img_path)
        size = (img.shape[1], img.shape[0])

        rect_camera_matrix = cv2.getOptimalNewCameraMatrix(mtx, dist, size, 0.0)[0]
        map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, rect_camera_matrix, size, cv2.CV_32FC1)

        undistorted_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        img_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, img_name), undistorted_img)


def stitch_images(start, end):
    """
    Create a panorama by stitching images sequentially.
    """
    img = cv2.imread(f'stitching_undistorted/img{start}.png')
    for i in range(start, end):
        H = get_homography(i, i + 1)
        next_img = cv2.imread(f'stitching_undistorted/img{i + 1}.png')
        img = get_panorama(next_img, img, H)

    return crop_img(img)


# Main Workflow
if __name__ == "__main__":
    # Load calibration images
    calibration_images_path = os.path.join(os.getcwd(), "calibration", "img*.png")
    calibration_images = glob.glob(calibration_images_path)

    # Define ArUco board for calibration
    marker_size, marker_gap = 168, 70
    step_size = marker_size + marker_gap

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
    mapping = {29: 1, 28: 0, 23: 2, 18: 4, 24: 3, 19: 5}

    aruco_board = np.array([
        [x * step_size, y * step_size, 0]
        for x in range(3) for y in range(2)
    ], dtype=np.float32)

    # Calibrate camera
    _, mtx, dist, _, _ = calibrate_camera(calibration_images, aruco_board, mapping)

    # Undistort images for stitching
    stitching_images_path = os.path.join(os.getcwd(), "stitching", "img*.png")
    stitching_images = glob.glob(stitching_images_path)
    undistort_images(stitching_images, mtx, dist, "stitching_undistorted")

    # Create panoramas
    panorama = stitch_images(4, 8)
    cv2.imwrite("panorama.png", panorama)

    print("Panorama stitching completed.")
