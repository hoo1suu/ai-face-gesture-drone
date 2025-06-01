import configargparse
import cv2
import cv2 as cv
from djitellopy import Tello
from gestures.tello_gesture_controller import TelloGestureController
from gestures.gesture_recognition import GestureRecognition, GestureBuffer
from utils import CvFpsCalc
import threading
import numpy as np
import os
import mediapipe as mp

# 얼굴 인식 모델 로드
script_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')
face_cascade = cv.CascadeClassifier(cascade_path)

def get_args():
    print('## Reading configuration ##')
    parser = configargparse.ArgParser(default_config_files=['config.txt'])

    parser.add_argument('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    parser.add_argument("--device", type=int)
    parser.add_argument("--width", help='cap width', type=int)
    parser.add_argument("--height", help='cap height', type=int)
    parser.add_argument('--use_static_image_mode', action='store_true', help='True if running on photos')
    parser.add_argument("--min_detection_confidence", help='min_detection_confidence', type=float)
    parser.add_argument("--min_tracking_confidence", help='min_tracking_confidence', type=float)
    parser.add_argument("--buffer_len", help='Length of gesture buffer', type=int)

    args = parser.parse_args()
    return args

def findFace(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imgGray, 1.1, 6)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        myFaceListArea.append(area)
        myFaceListC.append([cx, cy])

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

def trackFace(myDrone, info, w, pid, pError):
    error = info[0][0] - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    print(f"Tracking speed: {speed}")
    if info[0][0] != 0:
        myDrone.yaw_velocity = speed
    else:
        myDrone.for_back_velocity = 0
        myDrone.left_right_velocity = 0
        myDrone.up_down_velocity = 0
        myDrone.yaw_velocity = 0
        error = 0

    if myDrone.send_rc_control:
        myDrone.send_rc_control(myDrone.left_right_velocity,
                                myDrone.for_back_velocity,
                                myDrone.up_down_velocity,
                                myDrone.yaw_velocity)
    return error

def main():
    # Argument parsing
    args = get_args()
    WRITE_CONTROL = False
    in_flight = False
    w, h = args.width, args.height
    pid = [0.4, 0.4, 0]
    pError = 0

    # Camera preparation
    tello = Tello()
    tello.connect()
    tello.streamon()
    cap = tello.get_frame_read()

    # Add necessary velocity attributes to Tello object
    tello.for_back_velocity = 0
    tello.left_right_velocity = 0
    tello.up_down_velocity = 0
    tello.yaw_velocity = 0

    # Init Tello Controllers
    gesture_controller = TelloGestureController(tello)
    gesture_detector = GestureRecognition(args.use_static_image_mode, args.min_detection_confidence,
                                          args.min_tracking_confidence)
    gesture_buffer = GestureBuffer(buffer_len=args.buffer_len)

    def tello_control(gesture_controller):
        gesture_controller.gesture_control(gesture_buffer)

    def tello_battery(tello):
        nonlocal battery_status
        try:
            battery_status = tello.get_battery()
            battery_status = int(battery_status)
        except Exception as e:
            battery_status = -1
            print(f"Battery status error: {e}")

    # FPS Measurement
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    # --------- 모드 관련 변수 초기화 ---------
    mode = 'gesture'  # 'gesture', 'face', 'manual'
    number = -1
    battery_status = -1
    pError = 0

    while True:
        fps = cv_fps_calc.get()
        key = cv.waitKey(1) & 0xff

        # ----------- 모드 전환 ---------------
        if key == 27:  # ESC 종료
            break
        elif key == 32:  # Space - 이륙/정지
            if not in_flight:
                if battery_status == -1:
                    tello_battery(tello)
                if battery_status > 20:
                    try:
                        tello.takeoff()
                        in_flight = True
                        print("Takeoff!")
                    except Exception as e:
                        print(f"Takeoff error: {e}")
                else:
                    print("Battery is too low for takeoff.")
            else:
                tello.send_rc_control(0, 0, 0, 0)
                print("Hovering...")

        elif key == ord('g'):  # 제스처 모드로 전환
            mode = 'gesture'
            print("모드: 제스처 인식")
        elif key == ord('t'):  # 얼굴 추적 모드로 전환
            mode = 'face'
            pError = 0
            print("모드: 얼굴 추적")
        elif key == ord('m'):  # 키보드 수동 조종 모드
            mode = 'manual'
            print("모드: 키보드 수동조작 (WASD/R/F/E/Q)")

        # ------------ 영상 프레임 획득 -------------
        image = cap.frame
        image = cv2.cvtColor(image, cv.COLOR_RGB2BGR)
        debug_image = image.copy()

        # ----------- 모드별 동작 처리 -------------
        if mode == 'face':
            debug_image, info = findFace(image)
            pError = trackFace(tello, info, w, pid, pError)

        elif mode == 'gesture':
            gesture_id, results = None, None
            debug_image, gesture_id, results = gesture_detector.recognize(image, number, mode)
            gesture_buffer.add_gesture(gesture_id)
            # 손 랜드마크 그리기
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        debug_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            # 제스처 기반 스레드 컨트롤
            threading.Thread(target=tello_control, args=(gesture_controller,)).start()
        elif mode == 'manual':
            # 키보드로 직접 조종
            # W/S: 전진/후진, A/D: 좌/우 이동, R/F: 위/아래, Q/E: 좌/우 회전
            # 속도값
            speed = 30
            left_right_velocity = 0
            for_back_velocity = 0
            up_down_velocity = 0
            yaw_velocity = 0

            # 키보드 입력에 따라 값 세팅
            if key == ord('w'):
                for_back_velocity = speed
            elif key == ord('s'):
                for_back_velocity = -speed
            if key == ord('a'):
                left_right_velocity = -speed
            elif key == ord('d'):
                left_right_velocity = speed
            if key == ord('r'):
                up_down_velocity = speed
            elif key == ord('f'):
                up_down_velocity = -speed
            if key == ord('q'):
                yaw_velocity = -speed
            elif key == ord('e'):
                yaw_velocity = speed

            # 드론 조종 명령 보내기
            tello.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)

        # ----------- 배터리/영상 출력 -----------
        threading.Thread(target=tello_battery, args=(tello,)).start()
        cv.putText(debug_image, f"MODE: {mode.upper()}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(debug_image, "Battery: {}%".format(battery_status), (5, h - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('Tello Controller', debug_image)

    tello.land()
    tello.end()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
