import numpy as np


# Utility Functions
# Convert Mediapipe Landmark to 2D Coordinates
def unnormalize_point(landmark, shape):
    x = landmark.x
    y = landmark.y
    un_norm_x = int(x*shape[1])
    un_norm_y = int(y*shape[0])
    return np.array([un_norm_x, un_norm_y])


# Calculate Euclidean Distance of 2 points
def calc_distance_pts(point1, point2):
    distance = np.linalg.norm(point1-point2)
    return distance


# Calculate average Euclidean Distance of 2 list points:
def calc_distance_list(list1, list2):
    distance = np.mean((np.sum((list1-list2)**2, axis=1))**0.5)
    return distance


# Calibration of Roll, Yaw, Pitch
def calibrate_roll(degree, y1, y2):
    if y2 >= y1:
        result = degree
    else:
        result = -degree
    return result


# Action Inference Functions (Heuristic Based)
# 1. Eye_Closed
def are_eyes_close(landmarks, eye_left, eye_right, shape, threshold):
    eye_left_height = calc_distance_pts(unnormalize_point(landmarks[eye_left[0]], shape),
                                        unnormalize_point(landmarks[eye_left[2]], shape))
    eye_left_width = calc_distance_pts(unnormalize_point(landmarks[eye_left[1]], shape),
                                       unnormalize_point(landmarks[eye_left[3]], shape))

    eye_right_height = calc_distance_pts(unnormalize_point(landmarks[eye_right[0]], shape),
                                         unnormalize_point(landmarks[eye_right[2]], shape))
    eye_right_width = calc_distance_pts(unnormalize_point(landmarks[eye_right[1]], shape),
                                        unnormalize_point(landmarks[eye_right[3]], shape))

    if eye_left_height < eye_left_width * threshold and eye_right_height < eye_right_width * threshold:
        result = 1
    else:
        result = 0

    return result


# 2. Mouth Opened
def are_mouth_open(landmarks, mouth_upper_top, mouth_upper_bottom, mouth_lower_top, mouth_lower_bottom, shape,
                   threshold):
    unnorm_mouth_upper_top = np.array([unnormalize_point(landmarks[x], shape) for x in mouth_upper_top])
    unnorm_mouth_upper_bottom = np.array([unnormalize_point(landmarks[x], shape) for x in mouth_upper_bottom])
    unnorm_mouth_lower_top = np.array([unnormalize_point(landmarks[x], shape) for x in mouth_lower_top])
    unnorm_mouth_lower_bottom = np.array([unnormalize_point(landmarks[x], shape) for x in mouth_lower_bottom])

    lip_upper_avg_height = calc_distance_list(unnorm_mouth_upper_top, unnorm_mouth_upper_bottom)
    lip_lower_avg_height = calc_distance_list(unnorm_mouth_lower_top, unnorm_mouth_lower_bottom)

    lip_open_space_avg_height = calc_distance_list(unnorm_mouth_upper_bottom, unnorm_mouth_lower_top)

    if lip_open_space_avg_height > (lip_upper_avg_height + lip_lower_avg_height) * threshold:
        result = 1
    else:
        result = 0
    return result


# 3. Roll Left or Right
def roll_calc(landmarks, cheek_left, cheek_right, shape):
    unnorm_cheek_left = unnormalize_point(landmarks[cheek_left[0]], shape)
    unnorm_cheek_right = unnormalize_point(landmarks[cheek_right[0]], shape)

    x1 = unnorm_cheek_left[0]
    y1 = unnorm_cheek_left[1]
    x2 = unnorm_cheek_right[0]
    y2 = unnorm_cheek_right[1]

    height = abs(y2 - y1)
    width = abs(x2 - x1)

    degree = np.arctan(height / width) * 180 / np.pi
    return degree, y1, y2


def roll_trigger(degree, y1, y2, neutral, threshold):
    if degree - neutral > threshold and y2 > y1:
        result = 1
    elif degree + neutral > threshold and y2 <= y1:
        result = -1
    else:
        result = 0
    return result


# 4. Yaw Left or Right (neutral ratio basis = left/right)
def yaw_calc(landmarks, eye_tip_left, nose_bridge, eye_tip_right, shape):
    left2mid_dist = calc_distance_pts(unnormalize_point(landmarks[eye_tip_left[0]], shape),
                                      unnormalize_point(landmarks[nose_bridge[0]], shape))
    right2mid_dist = calc_distance_pts(unnormalize_point(landmarks[eye_tip_right[0]], shape),
                                       unnormalize_point(landmarks[nose_bridge[0]], shape))
    return right2mid_dist, left2mid_dist


def yaw_trigger(left2mid_dist, right2mid_dist, neutral, threshold):
    if right2mid_dist > left2mid_dist * threshold / neutral:
        result = 1
    elif left2mid_dist > right2mid_dist * threshold * neutral:
        result = -1
    else:
        result = 0
    return result


# 5. Pitch Up or Down (neutral ratio basis = upper_dist/lower_dist)
def pitch_calc(landmarks, forehead_upper, forehead_lower, chin_upper, chin_lower, shape):
    y_forehead_upper = (unnormalize_point(landmarks[forehead_upper[0]], shape)[1] +
                        unnormalize_point(landmarks[forehead_upper[1]], shape)[1]) / 2
    y_forehead_lower = (unnormalize_point(landmarks[forehead_lower[0]], shape)[1] +
                        unnormalize_point(landmarks[forehead_lower[1]], shape)[1]) / 2
    y_chin_upper = (unnormalize_point(landmarks[chin_upper[0]], shape)[1] +
                    unnormalize_point(landmarks[chin_upper[1]], shape)[1]) / 2
    y_chin_lower = (unnormalize_point(landmarks[chin_lower[0]], shape)[1] +
                    unnormalize_point(landmarks[chin_lower[1]], shape)[1]) / 2

    upper_dist = abs(y_forehead_upper - y_forehead_lower)
    lower_dist = abs(y_chin_upper - y_chin_lower)
    return upper_dist, lower_dist


def pitch_trigger(upper_dist, lower_dist, neutral, threshold):
    if upper_dist > lower_dist * threshold * neutral:
        result = 1
    elif lower_dist > upper_dist * threshold / neutral:
        result = -1
    else:
        result = 0
    return result
