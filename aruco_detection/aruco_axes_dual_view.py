import cv2
import numpy as np

# 설정값
DICT_ID = cv2.aruco.DICT_5X5_1000   # 사용 딕셔너리
MARKER_LENGTH_M = 0.05              # 마커 한 변 길이(미터)
CAM_INDEX = 0                       # 카메라 인덱스


def draw_pretty_axes(img, camera_matrix, dist_coeffs, rvec, tvec,
                     axis_len=0.06, thickness=3):
    axis_3d = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len],
    ]).reshape(-1, 3)

    pts2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    o, x, y, z = [tuple(p.ravel().astype(int)) for p in pts2d]

    cv2.line(img, o, x, (0, 0, 255), thickness, cv2.LINE_AA)   # X: R
    cv2.line(img, o, y, (0, 255, 0), thickness, cv2.LINE_AA)   # Y: G
    cv2.line(img, o, z, (255, 0, 0), thickness, cv2.LINE_AA)   # Z: B

    for p, c, label in [(x,(0,0,255),'X'), (y,(0,255,0),'Y'), (z,(255,0,0),'Z')]:
        cv2.circle(img, p, 5, c, -1, cv2.LINE_AA)
        cv2.putText(img, label, (p[0]+6, p[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)
    cv2.circle(img, o, 4, (255,255,255), -1, cv2.LINE_AA)


def get_intrinsics_rough(frame):
    h, w = frame.shape[:2]
    f = max(w, h) * 1.2
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print('카메라를 열 수 없습니다.')
        return

    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    cv2.namedWindow('Pretty Axes', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Default Axes', cv2.WINDOW_NORMAL)

    # 캘리브레이션 미제공 시 첫 프레임으로 대략 추정
    camera_matrix = None
    dist_coeffs = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if camera_matrix is None:
            camera_matrix, dist_coeffs = get_intrinsics_rough(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        pretty = frame.copy()
        default = frame.copy()

        if ids is not None and len(ids) > 0:
            # draw markers on both views
            cv2.aruco.drawDetectedMarkers(pretty, corners, ids)
            cv2.aruco.drawDetectedMarkers(default, corners, ids)

            # 포즈 추정
            rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs
            )
            for i in range(len(ids)):
                # 커스텀 축
                draw_pretty_axes(pretty, camera_matrix, dist_coeffs, rvecs[i], tvecs[i],
                                  axis_len=MARKER_LENGTH_M * 0.8, thickness=3)
                # 기본 축
                if hasattr(cv2, 'drawFrameAxes'):
                    cv2.drawFrameAxes(default, camera_matrix, dist_coeffs, rvecs[i], tvecs[i],
                                       MARKER_LENGTH_M * 0.8)
                elif hasattr(cv2.aruco, 'drawAxis'):
                    cv2.aruco.drawAxis(default, camera_matrix, dist_coeffs, rvecs[i], tvecs[i],
                                        MARKER_LENGTH_M * 0.8)

        cv2.imshow('Pretty Axes', pretty)
        cv2.imshow('Default Axes', default)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
