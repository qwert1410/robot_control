"""
Stub for homework 2
"""

import time
import random
import numpy as np
import mujoco
from mujoco import viewer

import numpy as np
import cv2
from numpy.typing import NDArray


TASK_ID = 3

world_xml_path = f"car_{TASK_ID}.xml"
model = mujoco.MjModel.from_xml_path(world_xml_path)
renderer = mujoco.Renderer(model, height=480, width=640)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)


def sim_step(
    n_steps: int, /, view=True, rendering_speed = 10, **controls: float
) -> NDArray[np.uint8]:
    """A wrapper around `mujoco.mj_step` to advance the simulation held in
    the `data` and return a photo from the dash camera installed in the car.

    Args:
        n_steps: The number of simulation steps to take.
        view: Whether to render the simulation.
        rendering_speed: The speed of rendering. Higher values speed up the rendering.
        controls: A mapping of control names to their values.
        Note that the control names depend on the XML file.

    Returns:
        A photo from the dash camera at the end of the simulation steps.

    Examples:
        # Advance the simulation by 100 steps.
        sim_step(100)

        # Move the car forward by 0.1 units and advance the simulation by 100 steps.
        sim_step(100, **{"forward": 0.1})

        # Rotate the dash cam by 0.5 radians and advance the simulation by 100 steps.
        sim_step(100, **{"dash cam rotate": 0.5})
    """

    for control_name, value in controls.items():
        data.actuator(control_name).ctrl = value

    for _ in range(n_steps):
        step_start = time.time()
        mujoco.mj_step(model, data)
        if view:
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step / rendering_speed)

    renderer.update_scene(data=data, camera="dash cam")
    img = renderer.render()
    return img



# TODO: add addditional functions/classes for task 1 if needed
def process_image(img):
    """Process the image to find the red ball."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_red1 = np.array([0, 50, 50]) 
    upper_red1 = np.array([10, 255, 255]) 
    lower_red2 = np.array([170, 240, 240]) 
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = cv2.contourArea(largest_contour)
            return cx, cy, area, mask  
    return None, None, None, mask
# /TODO


def task_1():
    steps = random.randint(0, 2000)
    controls = {"forward": 0, "turn": 0.1}
    img = sim_step(steps, view=False, **controls)
    cv2.imshow('', img)
    # TODO: Change the lines below.
    # For car control, you can use only sim_step function
    img_center = img.shape[1] // 2  

    while True:       
        cx, cy, area, mask = process_image(img)

        if cx is not None:
            if area > 6000:  
                break
            while abs(cx - img_center) > 20:
                turn_rate = -0.05 * (cx - img_center)
                img = sim_step(5, view=True, forward=0.0, turn=np.clip(turn_rate, -0.2, 0.2))
                cx, cy, area, mask = process_image(img)

            img = sim_step(50, view=True, forward=1.0, turn=0.0)

        else:
            img = sim_step(10, view=True, forward=0.0, turn=0.2)  
    # /TODO

# TODO: add addditional functions/classes for task 2 if needed
def calculate_wall_coverage(img):
    """Calculate the percentage of the image occupied by green or blue walls."""
    emergency_points = img[-(img.shape[0]//4)+10:-(img.shape[0]//4)+20, (img.shape[1]//2-60):(img.shape[1]//2+60), :].copy()
    np.save('emergency_points', emergency_points)
    turning_points = img[-(img.shape[0]//4)-10:-(img.shape[0]//4), (img.shape[1]//2-100):(img.shape[1]//2+100), :].copy()

    hsv_emergency = cv2.cvtColor(emergency_points, cv2.COLOR_RGB2HSV)
    hsv_turning = cv2.cvtColor(turning_points, cv2.COLOR_RGB2HSV)

    green_lower = np.array([50, 50, 50])
    green_upper = np.array([70, 255, 255])
    blue_lower = np.array([120, 50, 50])
    blue_upper = np.array([150, 255, 255])
    
    green_mask_emergency = cv2.inRange(hsv_emergency, green_lower, green_upper)
    blue_mask_emergency = cv2.inRange(hsv_emergency, blue_lower, blue_upper)

    emergency_pixels = np.sum(green_mask_emergency > 0) + np.sum(blue_mask_emergency > 0)
    
    green_mask_turning = cv2.inRange(hsv_turning, green_lower, green_upper)
    blue_mask_turning = cv2.inRange(hsv_turning, blue_lower, blue_upper)

    left_pixels = np.sum(green_mask_turning[:, :green_mask_turning.shape[1] // 4] > 0) + np.sum(blue_mask_turning[:, :blue_mask_turning.shape[1] // 4] > 0)
    right_pixels = np.sum(green_mask_turning[:, -green_mask_turning.shape[1] // 4:] > 0) + np.sum(blue_mask_turning[:, -blue_mask_turning.shape[1] // 4:] > 0)
    turning_pixels = left_pixels + right_pixels

    return turning_pixels, emergency_pixels
# /TODO

def task_2():
    speed = random.uniform(-0.3, 0.3)
    turn = random.uniform(-0.2, 0.2)
    controls = {"forward": speed, "turn": turn}
    img = sim_step(1000, view=True, **controls)
    np.save('img_before', img)
    # TODO: Change the lines below.
    # For car control, you can use only sim_step function
    counter = 0
    while True:
        turning_pixels, emergency_pixels = calculate_wall_coverage(img)
        if emergency_pixels:
            img = sim_step(20, view=True, forward=-0.3, turn=0.0)
            counter = 0

        elif turning_pixels:
            img = sim_step(20, view=True, forward=0.0, turn=0.2)
            counter = 0
        else:
            img = sim_step(20, view=True, forward=0.3, turn=0.0)
            counter += 1

        if counter == 200:
            task_1()
            break
        
    # /TODO



def ball_is_close() -> bool:
    """Checks if the ball is close to the car."""
    ball_pos = data.body("target-ball").xpos
    car_pos = data.body("dash cam").xpos
    print(car_pos, ball_pos)
    print('Distance to ball', np.linalg.norm(ball_pos - car_pos))
    return np.linalg.norm(ball_pos - car_pos) < 0.2


def ball_grab() -> bool:
    """Checks if the ball is inside the gripper."""
    print(data.body("target-ball").xpos[2])
    return data.body("target-ball").xpos[2] > 0.1


def teleport_by(x: float, y: float) -> None:
    data.qpos[0] += x
    data.qpos[1] += y
    sim_step(10, **{"dash cam rotate": 0})


def get_dash_camera_intrinsics():
    '''
    Returns the intrinsic matrix and distortion coefficients of the camera.
    '''
    h = 480
    w = 640
    o_x = w / 2
    o_y = h / 2
    fovy = 90
    f = h / (2 * np.tan(fovy * np.pi / 360))
    intrinsic_matrix = np.array([[-f, 0, o_x], [0, f, o_y], [0, 0, 1]])
    distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # no distortion

    return intrinsic_matrix, distortion_coefficients


# TODO: add addditional functions/classes for task 3 if needed
camera_angle = 0
def estimatePoseSingleMarkers(corners, ids, cameraMatrix, distCoeffs):
    corners_by_marker_id = {
        0: [
            (-0.275, -0.280, 0.05),  # Back-Right-Bottom (Marker 0, Right face)
            (-0.275, -0.280, 0.01),  # Front-Right-Bottom (Marker 0, Right face)
            (-0.275, -0.320, 0.01),  # Front-Right-Top (Marker 0, Right face)
            (-0.275, -0.320, 0.05)   # Back-Right-Top (Marker 0, Right face)
        ],
        1: [
            (-0.32, -0.275, 0.01),  # Back-Left-Bottom (Marker 5, Back face)
            (-0.28, -0.275, 0.01),  # Back-Right-Bottom (Marker 5, Back face)
            (-0.28, -0.275, 0.05),  # Back-Right-Top (Marker 5, Back face)
            (-0.32, -0.275, 0.05)   # Back-Left-Top (Marker 5, Back face)
        ]
    }
    marker_points = []
    marker_points.append(corners_by_marker_id[ids[0]])
    marker_points.append(corners_by_marker_id[ids[1]])
    marker_points = np.array(marker_points, dtype='float32')
    marker_points = marker_points.reshape(-1, 3)
    
    corners = np.vstack(corners)
    corners = corners.reshape(-1, 2)

    _, rvecs, tvecs = cv2.solvePnP(marker_points, corners, cameraMatrix, distCoeffs)
    return rvecs, tvecs

def teleport_to_ball(img):
    global camera_angle
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    detectorParams = cv2.aruco.DetectorParameters()
    detectorParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    detector = cv2.aruco.ArucoDetector(aruco_dict, detectorParams)
    intrinsic_matrix, dist_coeffs = get_dash_camera_intrinsics()

    corners, ids, _ = detector.detectMarkers(img)
    
    while ids is None:
        img = sim_step(150, **{"dash cam rotate": -1})
        img = sim_step(1, **{"dash cam rotate": 0})
        camera_angle += 1
        corners, ids, _ = detector.detectMarkers(img)

    while len(ids.flatten()) < 2:
        if ids.flatten()[0] == 0:
            teleport_by(0., 0.05)
        else:
            teleport_by(0.05, 0)
            
        img = sim_step(1, **{"dash cam rotate": 0})
        corners, ids, _ = detector.detectMarkers(img)

        while ids is None:
            img = sim_step(150, **{"dash cam rotate": -1})
            img = sim_step(1, **{"dash cam rotate": 0})
            camera_angle += 1
            corners, ids, _ = detector.detectMarkers(img)

    ids = ids.flatten()

    rvecs, tvecs = estimatePoseSingleMarkers(
        corners, ids,
        cameraMatrix=intrinsic_matrix,
        distCoeffs=dist_coeffs
    )
    
    R, _ = cv2.Rodrigues(rvecs)
    camera_position = -np.dot(R.T, tvecs)

    teleport_by(0.9 - camera_position[0], 1.9 - camera_position[1])

def locate_red_ball(img):        
    red_lower = np.array([100, 100, 100])
    red_upper = np.array([150, 255, 255])

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, red_lower, red_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    while len(contours) == 0:
        img = sim_step(50, **{"turn": -0.2})
        img = sim_step(1, **{"turn": 0.0})
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, red_lower, red_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)
    ball_center = cv2.minEnclosingCircle(largest_contour)[0]
    red_area = cv2.countNonZero(mask)
    return ball_center, red_area

def get_the_ball(img):
    global camera_angle
    img = sim_step(1000, **{'lift':1})

    for _ in range(camera_angle):
        img = sim_step(150, **{"dash cam rotate": 1})
        img = sim_step(1, **{"dash cam rotate": 0})

    ball_center, red_area = locate_red_ball(img)
    middle_point = img.shape[1] // 2

    img = sim_step(300, **{'forward':-0.5, 'turn':0})
    img = sim_step(1, **{'forward':0.0, 'turn':0})

    while True:
        if abs(ball_center[0] - middle_point) > 10:
            turn = 0.1 * np.sign(middle_point - ball_center[0])
            img = sim_step(20, **{'turn':turn})
            ball_center, red_area = locate_red_ball(img)

        elif red_area > 8500:
            img = sim_step(20, **{'forward':-0.1, 'turn':0})
            ball_center, red_area = locate_red_ball(img)

        elif red_area < 8000:
            img = sim_step(20, **{'forward':0.1, 'turn':0})
            ball_center, red_area = locate_red_ball(img)
        else:
            break
    
    img = sim_step(1000, **{'lift':-1, 'forward':0, 'turn':0})
    img = sim_step(1000, **{'trapdoor close/open':1})
    img = sim_step(1000, **{'lift':1})
# /TODO

def task_3():
    start_x = random.uniform(-0.2, 0.2)
    start_y = random.uniform(0, 0.2)
    img = sim_step(2000, **{"lift": 1})
    teleport_by(start_x, start_y)
    # TODO: Get to the ball
    #  - use the dash camera and ArUco markers to precisely locate the car
    #  - move the car to the ball using teleport_by function
    # img = sim_step(1, **{"dash cam rotate": 0, "lift": 0})

    teleport_to_ball(img)
    time.sleep(4)
    # /TODO
    assert ball_is_close()

    # TODO: Grab the ball
    # - the car should be already close to the ball
    # - use the gripper to grab the ball
    # - you can mo ve the car as well if you need to
    img = sim_step(1, **{"dash cam rotate": 0})
    get_the_ball(img)
    # /TODO

    assert ball_grab()


if __name__ == "__main__":
    print(f"Running TASK_ID {TASK_ID}")
    if TASK_ID == 1:
        task_1()
    elif TASK_ID == 2:
        task_2()
    elif TASK_ID == 3:
        task_3()
    else:
        raise ValueError("Unknown TASK_ID")
