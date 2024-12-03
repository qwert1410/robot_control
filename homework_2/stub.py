"""
Stub for homework 2
"""

import time
import random
import numpy as np
import mujoco
from mujoco import viewer
import math

import numpy as np
import cv2
from numpy.typing import NDArray


TASK_ID = 2


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
    lower_red2 = np.array([170, 50, 50]) 
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

    for _ in range(1500):        
        cx, cy, area, mask = process_image(img)

        if cx is not None:
            if area > 5000:  
                print("Stopping: Ball is close.")
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
THRESHOLD_PERCENTAGE = 10  # Percentage of the view covered by a wall to trigger a turn

def calculate_wall_coverage(img):
    """Calculate the percentage of the image occupied by green or blue walls."""
    img = img[:, (img.shape[1] // 2 - 5):(img.shape[1] // 2 + 5), :]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    green_lower = np.array([50, 50, 50])
    green_upper = np.array([70, 255, 255])
    blue_lower = np.array([110, 50, 50])
    blue_upper = np.array([130, 255, 255])

    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    total_pixels = img.shape[0] * img.shape[1]


    left_green_pixels = np.sum(green_mask[:, :green_mask.shape[1] // 2, :] > 0) / total_pixels * 100
    right_green_pixels = np.sum(green_mask[:, green_mask.shape[1] // 2:, :] > 0) / total_pixels * 100
    left_blue_pixels = np.sum(blue_mask[:, :blue_mask.shape[1] // 2, :] > 0) / total_pixels * 100
    right_blue_pixels = np.sum(blue_mask[:, blue_mask.shape[1] // 2:, :] > 0) / total_pixels * 100

    return left_green_pixels, right_green_pixels, left_blue_pixels, right_blue_pixels
# /TODO

def task_2():
    speed = random.uniform(-0.3, 0.3)
    turn = random.uniform(-0.2, 0.2)
    controls = {"forward": speed, "turn": turn}
    img = sim_step(1000, view=True, **controls)

    # TODO: Change the lines below.
    # For car control, you can use only sim_step function
    while True:
        green_percentage, blue_percentage = calculate_wall_coverage(img)
        # print(green_percentage, blue_percentage)

        if green_percentage > THRESHOLD_PERCENTAGE:
            img = sim_step(3, view=True, forward=0.0, turn=0.1)  
            print('Green wall!')

        elif blue_percentage > THRESHOLD_PERCENTAGE:
            img = sim_step(3, view=True, forward=0.0, turn=0.1)  
            print('Blue wall!')
            
        elif green_percentage + blue_percentage > THRESHOLD_PERCENTAGE * 1.5:
            img = sim_step(3, view=True, forward=0.0, turn=0.1)  
            print('Corner!')
        
        else:
            img = sim_step(3, view=True, forward=0.1, turn=0.0)
    # /TODO



def ball_is_close() -> bool:
    """Checks if the ball is close to the car."""
    ball_pos = data.body("target-ball").xpos
    car_pos = data.body("dash cam").xpos
    print(car_pos, ball_pos)
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
# /TODO


def task_3():
    start_x = random.uniform(-0.2, 0.2)
    start_y = random.uniform(0, 0.2)
    teleport_by(start_x, start_y)

    # TODO: Get to the ball
    #  - use the dash camera and ArUco markers to precisely locate the car
    #  - move the car to the ball using teleport_by function

    time.sleep(2)
    x_dest = random.uniform(-0.2, 0.2)
    y_dest = 1 + random.uniform(-0.2, 0.2)

    teleport_by(x_dest, y_dest)
    time.sleep(2)

    # /TODO

    assert ball_is_close()

    # TODO: Grab the ball
    # - the car should be already close to the ball
    # - use the gripper to grab the ball
    # - you can move the car as well if you need to
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
