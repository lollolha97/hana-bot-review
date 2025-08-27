import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def draw_pretty_axes(img, camera_matrix, dist_coeffs, rvec, tvec,
                     axis_len=0.06, thickness=3):
    # 3D 축 점 (원점 + x,y,z 끝점)
    axis_3d = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len],
    ]).reshape(-1, 3)

    # 2D로 투영
    pts2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    o, x, y, z = [tuple(p.ravel().astype(int)) for p in pts2d]

    # 축 그리기
    cv2.line(img, o, x, (0, 0, 255), thickness, cv2.LINE_AA)   # X: 빨강
    cv2.line(img, o, y, (0, 255, 0), thickness, cv2.LINE_AA)   # Y: 초록
    cv2.line(img, o, z, (255, 0, 0), thickness, cv2.LINE_AA)   # Z: 파랑

    # 끝점 강조
    for p, c, label in [(x,(0,0,255),'X'), (y,(0,255,0),'Y'), (z,(255,0,0),'Z')]:
        cv2.circle(img, p, 5, c, -1, cv2.LINE_AA)
        cv2.putText(img, label, (p[0]+6, p[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)

    # 원점도 표시
    cv2.circle(img, o, 4, (255,255,255), -1, cv2.LINE_AA)

# IMG_PATH = Path(__file__).parent / 'aruco.png'
# img = cv2.imread(str(IMG_PATH))
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# MacBook Air 카메라 경험적 추정값
camera_matrix = np.array([
    [1400, 0, 960],     # fx ≈ 1400 픽셀
    [0, 1400, 540],     # fy ≈ 1400 픽셀
    [0, 0, 1]
])
dist_coeffs = np.array([0.01, -0.01, 0, 0, 0])  # 매우 작은 왜곡

marker_size = 0.05  # 예: 5cm = 0.05m

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    corners, ids, rejected = detector.detectMarkers(gray)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

    for i, c in enumerate(corners):
        # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)
        draw_pretty_axes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i],
                 axis_len=0.05, thickness=3)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)


    cv2.resizeWindow('aruco', 640, 480)
    cv2.imshow('aruco', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()