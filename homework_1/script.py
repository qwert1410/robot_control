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
    
    transformed_image = np.zeros((output_size[1], output_size[0], 3), dtype=image.dtype)
    
    for y in range(output_size[1]):
        for x in range(output_size[0]):
            
            dest_coord = np.array([x, y, 1])
            
            source_coord = H_inv @ dest_coord
            source_x, source_y = source_coord[:2] / source_coord[2]
            
            source_x = int(round(source_x))
            source_y = int(round(source_y))
            
            if 0 <= source_x < image.shape[1] and 0 <= source_y < image.shape[0]:
                transformed_image[y, x] = image[source_y, source_x]

    return transformed_image

def compute_homography(points_src, points_dst):
    """
    Compute the homography matrix that maps points_src to points_dst.
    """
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


def calibrate_camera_board(images, aruco_board):
    """
    Calibrate the camera using ArUco boards.
    """
    mapping = {29: 1, 28: 0, 23: 2, 18: 4, 24: 3, 19: 5}
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

def calibrate_camera(images, aruco_marker):
    """
    Calibrate the camera using ArUco markers.
    """
    objpoints = []
    imgpoints = []

    for image_path in images:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            for marker in corners:
                points = marker.reshape(-1, 1, 2)
                points = points.astype('float32')
                imgpoints.append(points)
                objpoints.append(aruco_marker)

    return cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

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

def hand_stitch(): #We stithc together picture 5 and 6
    img1_points = np.array([
        [989,454], #left upper corner white
        [1052,405], #left lower corner white
        [1076,434], #right upper corner white
        [1012,485], #right lower corner white
        [486,264], #left upper corner purple
        [514,250], #left lower corner purple
        [534,280], #right upper corner purple
        [507,295], #right lower corner purple
    ])

    img2_points = np.array([
        [1100,463], #left upper corner white
        [1171,411], #left lower corner white
        [1200,444], #right upper corner white
        [1127,496], #right lower corner white
        [587,272], #left upper corner purple
        [609,257], #left lower corner purple
        [630,287], #right upper corner purple
        [605,301], #right lower corner purple
    ])
    img_1 = cv2.imread(f'stitching_undistorted/img6.png')
    img_2 = cv2.imread(f'stitching_undistorted/img5.png')

    H = compute_homography(img1_points, img2_points)
    img = get_panorama(img_1, img_2, H)
    cv2.imwrite(f'hand_stitched_56.png', img)

def get_seam(img1_nr, img2_nr):
    img_1 = cv2.imread(f'stitching_undistorted/img{img2_nr}.png')
    img_2 = cv2.imread(f'stitching_undistorted/img{img1_nr}.png')

    H = get_homography(img1_nr, img2_nr)
    img_2 = get_panorama(img_1, img_2, H, False)

    h, w, _ = img_1.shape

    diffs = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            if i == 0:
                diffs[i, j] = col_diff(img_1[i, j], img_2[i, j])
            elif j == 0:
                diffs[i, j] = 1e10
            elif img_2[i, j].max() == 0: #black spot
                diffs[i, j] = 1e10
            else:
                col_difference = col_diff(img_1[i, j], img_2[i, j])
                diffs[i, j] = np.min([diffs[i-1, j-1], diffs[i-1, j], diffs[i-1, j]])
                diffs[i, j] += col_difference

    joined_img = img_2.copy()

    current_w = diffs[-1, :].argmin()
    current_w

    seam = np.array([h-1, current_w])

    joined_img[h-1, :current_w, :] = img_1[h-1, :current_w, :] #left picture
    joined_img[h-1, current_w, :] = [255, 255, 255] #seam

    for i in range(h-2, -1, -1):
        current_w += diffs[i, (current_w-1):(current_w+2)].argmin() - 1
        seam = np.append(seam, [i, current_w])
        joined_img[i, :current_w, :] = img_1[i, :current_w, :] #left picture
        joined_img[i, current_w, :] = [255, 255, 255] #seam

    seam = seam.reshape(-1, 2)


    joined_img = crop_img(joined_img)
    cv2.imwrite(f'best_seam_{img1_nr}{img2_nr}.png', joined_img)


def create_panorama(start, end):
    """
    Create a panorama by stitching images sequentially.
    """
    img = cv2.imread(f'stitching_undistorted/img{start}.png')
    for i in range(start, end):
        H = get_homography(i, i + 1)
        next_img = cv2.imread(f'stitching_undistorted/img{i + 1}.png')
        img = get_panorama(next_img, img, H)

    return crop_img(img)


if __name__ == "__main__":
    calibration_images_path = os.path.join(os.getcwd(), "calibration", "img*.png")
    calibration_images = glob.glob(calibration_images_path)

    marker_size, marker_gap = 168, 70
    step_size = marker_size + marker_gap
    top_left = [0, 0, 0]

    aruco_marker = [
        [top_left[0], top_left[1] + marker_size, 0],
        top_left,
        [top_left[0] + marker_size, top_left[1], 0],         
        [top_left[0] + marker_size, top_left[1] + marker_size, 0],   
    ]

    aruco_board = []
    for x in range(3):
        for y in range(2):    
            top_left = [x * step_size, (y) * step_size, 0]
            
            corners_3d = [
                [top_left[0], top_left[1] + marker_size, 0],
                top_left,
                [top_left[0] + marker_size, top_left[1], 0],    
                [top_left[0] + marker_size, top_left[1] + marker_size, 0],             
            ]
            
            aruco_board.extend(corners_3d)
    aruco_board = np.array(aruco_board, dtype=np.float32)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

    #Task 1
    _, mtx_board, dist_board, _, _ = calibrate_camera_board(calibration_images, aruco_board)
    _, mtx, dist, _, _ = calibrate_camera(calibration_images, aruco_marker)

    stitching_images_path = os.path.join(os.getcwd(), "stitching", "img*.png")
    stitching_images = glob.glob(stitching_images_path)
    undistort_images(stitching_images, mtx, dist, "stitching_undistorted")

    #Task 2
    # This function is defined above

    #Task 3
    print('Testint the compute_homography function.')
    test_homography()
    
    #Task 4
    hand_stitch()

    #Task 5
    get_seam(7, 8)
    #Task 6
    stitched = create_panorama(7, 8)
    cv2.imwrite("stitched_image_78.png", stitched)
    #Task 7
    panorama = create_panorama(4, 8)
    cv2.imwrite("panorama.png", panorama)
