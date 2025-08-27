import cv2
from pathlib import Path

# 출력 파일 경로 (스크립트와 같은 폴더)
OUT_PATH = Path(__file__).parent / 'aruco_generated.png'

# 사용할 사전과 마커 설정 (간단 기본값)
DICT_ID = cv2.aruco.DICT_5X5_1000
MARKER_ID = 0
SIDE_PIXELS = 400  # 출력 이미지 한 변의 픽셀 수

# 사전 준비 (4.12 API)
dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)

# 마커 이미지 생성: 4.12의 generateImageMarker 우선, 없으면 drawMarker로 폴백
try:
    marker_img = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, SIDE_PIXELS)
except AttributeError:
    marker_img = cv2.aruco.drawMarker(dictionary, MARKER_ID, SIDE_PIXELS)

# 파일로 저장 및 화면 표시
cv2.imwrite(str(OUT_PATH), marker_img)
cv2.imshow('ArUco Marker', marker_img)
print(f"Saved: {OUT_PATH}")
print("아무 키나 누르면 창이 닫힙니다.")
cv2.waitKey(0)
cv2.destroyAllWindows()
