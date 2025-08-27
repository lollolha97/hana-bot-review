# ArUco Detection Project

ArUco 마커를 검출하고 위치를 추정하는 Python 프로젝트입니다.

## 요구사항

- **Python**: 3.8 이상
- **운영체제**: Windows, macOS, Linux 지원

## 설치 방법

1. 가상환경 생성 (권장):
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 또는
venv\Scripts\activate     # Windows
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용법

```bash
python detect_aruco.py
```

## 프로젝트 구조

```
aruco_detection/
├── detect_aruco.py      # 메인 ArUco 검출 코드
├── requirements.txt     # 필요한 패키지 목록
├── README.md           # 이 파일
└── venv/               # 가상환경 (생성됨)
```

## 개발 환경 설정

1. **가상환경 활성화**: `source venv/bin/activate`
2. **패키지 설치**: `pip install -r requirements.txt`
3. **코드 실행**: `python detect_aruco.py`

## 문제 해결

- OpenCV 설치 문제: `pip install opencv-python-headless` 시도
- 권한 문제: `sudo pip install` 또는 가상환경 사용
