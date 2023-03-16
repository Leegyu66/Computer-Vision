import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# For webcam input:
cap = cv2.VideoCapture(0)

# 이미지 불러오기
image_right_eye = cv2.imread('right_eye.png') # 100 X 100
image_left_eye = cv2.imread('left_eye.png') # 100 X 100
image_nose = cv2.imread('nose.png') # 100 X 100
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

                keypoints = detection.location_data.relative_keypoints
                right_eye = keypoints[0] # 오른쪽 눈
                left_eye = keypoints[1] # 왼쪽 눈
                nose_tip = keypoints[2]

                h, w, _ = image.shape # height, width, channel
                right_eye = (int(right_eye.x * w), int(right_eye.y * h))
                left_eye = (int(left_eye.x * w), int(left_eye.y * h))
                nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))

                # cv2.circle(image, right_eye, 50, (255, 0, 0), 10, cv2.LINE_AA) # 파란색
                # cv2.circle(image, left_eye, 50, (0, 255, 0), 10, cv2.LINE_AA)
                # cv2.circle(image, nose_tip, 75, (0, 255, 255), 10, cv2.LINE_AA)
                
                image[right_eye[1] - 50:right_eye[1] + 50, right_eye[0] - 50:right_eye[0] + 50] = image_right_eye
                image[left_eye[1] - 50:left_eye[1] + 50, left_eye[0] - 50:left_eye[0] + 50] = image_left_eye
                image[nose_tip[1] - 50:nose_tip[1] + 50, nose_tip[0] - 50:nose_tip[0] + 50] = image_nose


        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', cv2.resize(image, None, fx=0.7, fy=0.7))

        if cv2.waitKey(1) == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()