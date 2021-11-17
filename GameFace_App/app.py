import pandas as pd
import numpy as np
import cv2
import time
import winsound
import webbrowser
import pyautogui
import pygetwindow as gw
import mediapipe as mp

from utils import *

# Loading Config Files
config_file = 'configs.txt'
openconfig = open(config_file, 'r')
config_read = openconfig.readlines()
openconfig.close()
configs = [x.strip() for x in config_read]

# Extract Config Information

# Extract Landmarks Index
eye_left = np.array(np.array(configs)[np.array(['eye_left' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)
eye_right = np.array(np.array(configs)[np.array(['eye_right' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)
mouth_upper_top = np.array(np.array(configs)[np.array(['mouth_upper_top' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)
mouth_upper_bottom = np.array(np.array(configs)[np.array(['mouth_upper_bottom' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)
mouth_lower_top = np.array(np.array(configs)[np.array(['mouth_lower_top' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)
mouth_lower_bottom = np.array(np.array(configs)[np.array(['mouth_lower_bottom' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)
cheek_left = np.array(np.array(configs)[np.array(['cheek_left' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)
cheek_right = np.array(np.array(configs)[np.array(['cheek_right' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)
eye_tip_left = np.array(np.array(configs)[np.array(['eye_tip_left' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)
nose_bridge = np.array(np.array(configs)[np.array(['nose_bridge' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)
eye_tip_right = np.array(np.array(configs)[np.array(['eye_tip_right' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)
forehead_upper = np.array(np.array(configs)[np.array(['forehead_upper' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)
forehead_lower = np.array(np.array(configs)[np.array(['forehead_lower' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)
chin_upper = np.array(np.array(configs)[np.array(['chin_upper' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)
chin_lower = np.array(np.array(configs)[np.array(['chin_lower' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(int)

# Extract Action Inference Threshold
eye_thres = np.array(np.array(configs)[np.array(['eye_thres' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(float)[0]
mouth_thres = np.array(np.array(configs)[np.array(['mouth_thres' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(float)[0]
roll_thres = np.array(np.array(configs)[np.array(['roll_thres' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(float)[0]
yaw_thres = np.array(np.array(configs)[np.array(['yaw_thres' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(float)[0]
pitch_thres = np.array(np.array(configs)[np.array(['pitch_thres' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(float)[0]

# Extract Action Inference Labels
eye_lab = np.array([x.strip() for x in np.array(configs)[np.array(['eye_lab' in x for x in config_read])][0].split('=')[-1].strip().split(',')])
mouth_lab = np.array([x.strip() for x in np.array(configs)[np.array(['mouth_lab' in x for x in config_read])][0].split('=')[-1].strip().split(',')])
roll_lab = np.array([x.strip() for x in np.array(configs)[np.array(['roll_lab' in x for x in config_read])][0].split('=')[-1].strip().split(',')])
yaw_lab = np.array([x.strip() for x in np.array(configs)[np.array(['yaw_lab' in x for x in config_read])][0].split('=')[-1].strip().split(',')])
pitch_lab = np.array([x.strip() for x in np.array(configs)[np.array(['pitch_lab' in x for x in config_read])][0].split('=')[-1].strip().split(',')])

# Extract Action Controls
eyes_crtl = np.array(configs)[np.array(['eyes_crtl' in x for x in config_read])][0].split('=')[-1].strip()
mouth_crtl = np.array(configs)[np.array(['mouth_crtl' in x for x in config_read])][0].split('=')[-1].strip()
pitch_up_crtl = np.array(configs)[np.array(['pitch_up_crtl' in x for x in config_read])][0].split('=')[-1].strip()
pitch_down_crtl = np.array(configs)[np.array(['pitch_down_crtl' in x for x in config_read])][0].split('=')[-1].strip()
yaw_left_crtl = np.array(configs)[np.array(['yaw_left_crtl' in x for x in config_read])][0].split('=')[-1].strip()
yaw_right_crtl = np.array(configs)[np.array(['yaw_right_crtl' in x for x in config_read])][0].split('=')[-1].strip()
roll_left_crtl = np.array(configs)[np.array(['roll_left_crtl' in x for x in config_read])][0].split('=')[-1].strip()
roll_right_crtl = np.array(configs)[np.array(['roll_right_crtl' in x for x in config_read])][0].split('=')[-1].strip()

# Extract Action Time Holding Threshold
eye_close_hold = np.array(np.array(configs)[np.array(['eye_close_hold' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(float)[0]
mouth_open_hold = np.array(np.array(configs)[np.array(['mouth_open_hold' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(float)[0]
command_exec_thres = np.array(np.array(configs)[np.array(['command_exec_thres' in x for x in config_read])][0].split('=')[-1].strip().split(',')).astype(float)[0]

# Extract Target Webpage
webpage = np.array(configs)[np.array(['webpage' in x for x in config_read])][0].split('=')[-1].strip()


def runapp():
    # Initialize Webcam
    video_capture = cv2.VideoCapture(0)

    # Initialize Time Recording
    previous = time.time()

    # Initial Position
    neutral_roll = 0  # Neutral Angle
    neutral_yaw = 1  # Ratio of left to Right
    neutral_pitch = 1  # Ratio of Up to Down

    # Status Indicator
    stdby_mode = True
    play_mode = False
    pend_command = True
    eye_close_time = 0
    mouth_open_time = 0
    command_exec_time = 0
    prev_command = None
    point_annotation = True

    # Calibration Euler:
    cali_roll_list = np.array([])
    cali_yaw_list = np.array([])
    cali_pitch_list = np.array([])

    # Mediapipe FaceMesh Initialization
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize Browser
    webbrowser.open_new(webpage)
    time.sleep(1)
    window_browser = gw.getActiveWindow()
    pyautogui.hotkey('winleft', 'right')
    window_active = False

    # Initialize Controls
    first_launch = True

    while video_capture.isOpened():
        # Exit Condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Start reading Image from Webcam
        ret, frame = video_capture.read()

        # Exit Condition if no image capture
        if not ret:
            break

        # Preprocessing - Flipping Left Right
        frame = cv2.flip(frame, 1)
        shape = frame.shape

        # Update Current Mode
        if stdby_mode and not play_mode:
            cv2.putText(frame, 'Standby Mode', (545, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        elif not stdby_mode and play_mode:
            cv2.putText(frame, 'Play Mode', (569, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        # Detect Landmarks
        frame.flags.writeable = False
        faces = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame.flags.writeable = True
        if faces.multi_face_landmarks is not None:
            face_ind = True
            all_landmarks = faces.multi_face_landmarks[0].landmark
            cv2.putText(frame, 'Face Detected', (543, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            face_ind = False
            cv2.putText(frame, 'No Face Detected', (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                        cv2.LINE_AA)

        # Update Status of Pending Command or under Latency
        if face_ind and pend_command:
            cv2.putText(frame, 'Pending Command', (515, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        elif face_ind and not pend_command:
            cv2.putText(frame, 'Breathing Time', (539, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        # Infer Actions
        if face_ind:
            eye_close = are_eyes_close(all_landmarks, eye_left, eye_right, shape, eye_thres)
            mouth_open = are_mouth_open(all_landmarks, mouth_upper_top, mouth_upper_bottom, mouth_lower_top,
                                        mouth_lower_bottom, shape, mouth_thres)

            roll_degree, roll_y1, roll_y2 = roll_calc(all_landmarks, cheek_left, cheek_right,
                                                      shape)  # , neutral_roll, roll_thres)
            roll = roll_trigger(roll_degree, roll_y1, roll_y2, neutral_roll, roll_thres)

            yaw_right2mid, yaw_left2mid = yaw_calc(all_landmarks, eye_tip_left, nose_bridge, eye_tip_right,
                                                   shape)  # , neutral_yaw, yaw_thres)
            yaw = yaw_trigger(yaw_right2mid, yaw_left2mid, neutral_yaw, yaw_thres)

            pitch_upper_dist, pitch_lower_dist = pitch_calc(all_landmarks, forehead_upper, forehead_lower, chin_upper,
                                                            chin_lower, shape)
            pitch = pitch_trigger(pitch_upper_dist, pitch_lower_dist, neutral_pitch, pitch_thres)

            cv2.putText(frame, eye_lab[eye_close], (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, mouth_lab[mouth_open], (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, roll_lab[roll + 1], (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, yaw_lab[yaw + 1], (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, pitch_lab[pitch + 1], (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 1,
                        cv2.LINE_AA)

        # Annotate Key Facial Landmarks Used
        if face_ind and point_annotation:
            key_ldmrk = np.concatenate([eye_left, eye_right,
                                        mouth_upper_top, mouth_upper_bottom, mouth_lower_top, mouth_lower_bottom,
                                        cheek_left, cheek_right,
                                        eye_tip_left, nose_bridge, eye_tip_right,
                                        forehead_upper, forehead_lower, chin_upper, chin_lower])
            key_coord = np.array([unnormalize_point(all_landmarks[x], shape) for x in key_ldmrk])
            # Draw Eye Points
            for i in key_coord[:8]:
                cv2.circle(frame, i, 1, (0, 255, 0), 1)
            # Draw Mouth Points
            for i in key_coord[8:20]:
                cv2.circle(frame, i, 1, (0, 255, 0), 1)
            # Draw Cheek Line
            cv2.line(frame, key_coord[20], key_coord[21], (255, 255, 255), 1)
            # Draw Left Eye Tip to Nose Bridge
            cv2.line(frame, key_coord[22], key_coord[23], (255, 255, 255), 1)
            # Draw Right Eye Tip to Nose Bridge
            cv2.line(frame, key_coord[23], key_coord[24], (255, 255, 255), 1)
            # Draw ForeHead Upper
            cv2.line(frame, key_coord[25], key_coord[26], (255, 255, 255), 1)
            # Draw ForeHead Lower
            cv2.line(frame, key_coord[27], key_coord[28], (255, 255, 255), 1)
            # Draw Chin Upper
            cv2.line(frame, key_coord[29], key_coord[30], (255, 255, 255), 1)
            # Draw Chin Lower
            cv2.line(frame, key_coord[31], key_coord[32], (255, 255, 255), 1)

        # Enter Play Mode, Standby Mode -> Play Mode
        if face_ind and stdby_mode and not play_mode and pend_command:
            # Trigger Play Mode
            if eye_close == 1:
                cv2.putText(frame, 'Turning On...', (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                            cv2.LINE_AA)

            if eye_close_time == 0 and eye_close == 1:
                eye_close_time = time.time()
                cali_roll_list = np.append(cali_roll_list, np.array(calibrate_roll(roll_degree, roll_y1, roll_y2)))
                cali_yaw_list = np.append(cali_yaw_list, np.array([yaw_right2mid / yaw_left2mid]))
                cali_pitch_list = np.append(cali_pitch_list, np.array([pitch_upper_dist / pitch_lower_dist]))
                cv2.putText(frame, 'Hold Time  : {0:0.2f}'.format(round(time.time() - eye_close_time, 2)), (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            elif eye_close_time > 0 and eye_close == 1 and (time.time() - eye_close_time) >= eye_close_hold:
                cv2.putText(frame, 'Hold Time  : {0:0.2f}'.format(round(time.time() - eye_close_time, 2)), (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                stdby_mode = False
                play_mode = True
                eye_close_time = 0
                neutral_roll = np.mean(cali_roll_list)
                neutral_yaw = np.mean(cali_yaw_list)
                neutral_pitch = np.mean(cali_pitch_list)
                cali_roll_list = np.array([])
                cali_yaw_list = np.array([])
                cali_pitch_list = np.array([])
                winsound.Beep(440, 500)
                if first_launch:
                    pyautogui.press(mouth_crtl)
                    first_launch = False
                else:
                    pyautogui.press(eyes_crtl)

            elif eye_close_time > 0 and eye_close == 1 and (time.time() - eye_close_time) < eye_close_hold:
                cv2.putText(frame, 'Hold Time  : {0:0.2f}'.format(round(time.time() - eye_close_time, 2)), (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                cali_roll_list = np.append(cali_roll_list, np.array(calibrate_roll(roll_degree, roll_y1, roll_y2)))
                cali_yaw_list = np.append(cali_yaw_list, np.array([yaw_right2mid / yaw_left2mid]))
                cali_pitch_list = np.append(cali_pitch_list, np.array([pitch_upper_dist / pitch_lower_dist]))

            elif eye_close_time > 0 and eye_close == 0:
                eye_close_time = 0
                cali_roll_list = np.array([])
                cali_yaw_list = np.array([])
                cali_pitch_list = np.array([])

        # Exit Play Mode, Play Mode -> Standby Mode
        if face_ind and not stdby_mode and play_mode and pend_command:
            if eye_close == 1:
                cv2.putText(frame, 'Turning Off...', (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                            cv2.LINE_AA)
            if eye_close_time == 0 and eye_close == 1:
                eye_close_time = time.time()
                cv2.putText(frame, 'Hold Time  : {0:0.2f}'.format(round(time.time() - eye_close_time, 2)), (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            elif eye_close_time > 0 and eye_close == 1 and (time.time() - eye_close_time) >= eye_close_hold:
                cv2.putText(frame, 'Hold Time  : {0:0.2f}'.format(round(time.time() - eye_close_time, 2)), (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                stdby_mode = True
                play_mode = False
                eye_close_time = 0
                winsound.Beep(440, 500)
                pyautogui.press(eyes_crtl)
            elif eye_close_time > 0 and eye_close == 1 and (time.time() - eye_close_time) < eye_close_hold:
                cv2.putText(frame, 'Hold Time  : {0:0.2f}'.format(round(time.time() - eye_close_time, 2)), (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            elif eye_close_time > 0 and eye_close == 0:
                eye_close_time = 0

        # Execute Action
        if face_ind and not stdby_mode and play_mode and pend_command:
            if roll == 1:
                command_exec_time = time.time()
                prev_command = roll_lab[roll + 1]
                pend_command = False
                pyautogui.press(roll_right_crtl)
            elif roll == -1:
                command_exec_time = time.time()
                prev_command = roll_lab[roll + 1]
                pend_command = False
                pyautogui.press(roll_left_crtl)
            elif pitch == 1:
                command_exec_time = time.time()
                prev_command = pitch_lab[pitch + 1]
                pend_command = False
                pyautogui.press(pitch_up_crtl)
            elif pitch == -1:
                command_exec_time = time.time()
                prev_command = pitch_lab[pitch + 1]
                pend_command = False
                pyautogui.press(pitch_down_crtl)
            elif yaw == 1:
                command_exec_time = time.time()
                prev_command = yaw_lab[yaw + 1]
                pend_command = False
                pyautogui.press(yaw_right_crtl)
            elif yaw == -1:
                command_exec_time = time.time()
                prev_command = yaw_lab[yaw + 1]
                pend_command = False
                pyautogui.press(yaw_left_crtl)

        # Trigger for Mouth
        if face_ind and not stdby_mode and play_mode and pend_command:
            if mouth_open_time == 0 and mouth_open == 1:
                mouth_open_time = time.time()
                cv2.putText(frame, 'Hold Time  : {0:0.2f}'.format(round(time.time() - mouth_open_time, 2)), (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            elif mouth_open_time > 0 and mouth_open == 1 and (time.time() - mouth_open_time) >= mouth_open_hold:
                cv2.putText(frame, 'Hold Time  : {0:0.2f}'.format(round(time.time() - mouth_open_time, 2)), (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                mouth_open_time = 0
                pyautogui.press(mouth_crtl)
            elif mouth_open_time > 0 and mouth_open == 1 and (time.time() - mouth_open_time) < mouth_open_hold:
                cv2.putText(frame, 'Hold Time  : {0:0.2f}'.format(round(time.time() - mouth_open_time, 2)), (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            elif mouth_open_time > 0 and mouth_open == 0:
                mouth_open_time = 0

        # Breathing Period
        if face_ind and not stdby_mode and play_mode and not pend_command:
            if (time.time() - command_exec_time) >= command_exec_thres:
                command_exec_time = 0
                prev_command = None
                pend_command = True
            else:
                cv2.putText(frame, 'Action', (595, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f'{prev_command}', (570, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                            cv2.LINE_AA)

        # Refresh FPS
        new = time.time()
        f = int(1 / (new - previous))
        previous = new
        cv2.putText(frame, 'FPS  : {0:d}'.format(f), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1,
                    cv2.LINE_AA)

        # Showing Image Result
        winname = "Test"
        cv2.namedWindow(winname)
        if not window_active:
            cv2.moveWindow(winname, 300, 300)
        cv2.imshow(winname, frame)

        # Switching to application window
        if not window_active:
            window_active = True
            window_browser.activate()

    video_capture.release()
    cv2.destroyAllWindows()
