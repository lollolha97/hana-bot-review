import cv2
from collections import deque
import numpy as np

cap = cv2.VideoCapture(0)

tm = cv2.TickMeter()
tm.start()

# 최근 FPS 값을 저장할 버퍼 (최근 120프레임)
fps_history = deque(maxlen=120)

# 그래프 렌더링 설정
graph_width, graph_height = 400, 160
graph_margin = 30

def render_fps_graph(values):
    """FPS 히스토리를 간단한 라인 차트로 시각화하여 이미지로 반환한다."""
    canvas = np.full((graph_height, graph_width, 3), 20, dtype=np.uint8)

    if not values:
        cv2.putText(canvas, "FPS: --", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return canvas

    # y축 스케일링 (하한/상한 약간 여유)
    min_fps = max(0.0, min(values) - 2.0)
    max_fps = max(values) + 2.0
    if max_fps - min_fps < 1e-3:
        max_fps = min_fps + 1.0

    def to_point(idx, val, count):
        x0 = graph_margin
        x1 = graph_width - graph_margin
        y0 = graph_margin
        y1 = graph_height - graph_margin
        x = int(x0 + (x1 - x0) * (idx / max(1, count - 1)))
        norm = (val - min_fps) / (max_fps - min_fps)
        y = int(y1 - (y1 - y0) * norm)
        return (x, y)

    # 테두리 및 y축 레이블
    cv2.rectangle(canvas, (graph_margin, graph_margin), (graph_width - graph_margin, graph_height - graph_margin), (80, 80, 80), 1)
    mid_fps = 0.5 * (min_fps + max_fps)
    for v, ytxt in [(max_fps, graph_margin + 5), (mid_fps, graph_height // 2), (min_fps, graph_height - graph_margin - 5)]:
        txt = f"{v:.1f}"
        cv2.putText(canvas, txt, (5, int(ytxt)), cv2.FONT_HERSHEY_PLAIN, 1.0, (160, 160, 160), 1)

    # 최고/최소 FPS 값 찾기
    current_max = max(values)
    current_min = min(values)
    
    # 최고/최소값 위치에 점 찍기
    max_idx = values.index(current_max)
    min_idx = values.index(current_min)
    
    max_point = to_point(max_idx, current_max, len(values))
    min_point = to_point(min_idx, current_min, len(values))
    
    # 최고값 (빨간색 점)
    cv2.circle(canvas, max_point, 4, (0, 0, 255), -1)
    cv2.putText(canvas, f"MAX: {current_max:.1f}", (max_point[0] + 5, max_point[1] - 5), 
                cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 255), 1)
    
    # 최소값 (파란색 점)
    cv2.circle(canvas, min_point, 4, (255, 0, 0), -1)
    cv2.putText(canvas, f"MIN: {current_min:.1f}", (min_point[0] + 5, min_point[1] + 15), 
                cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 0, 0), 1)

    # 라인 그리기
    pts = [to_point(i, v, len(values)) for i, v in enumerate(values)]
    if len(pts) >= 2:
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], isClosed=False, color=(0, 220, 0), thickness=2)

    # 현재 FPS 텍스트
    cv2.putText(canvas, f"FPS: {values[-1]:.1f}", (graph_margin, graph_margin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    
    # 통계 정보 (우측 상단)
    stats_text = f"MAX: {current_max:.1f} | MIN: {current_min:.1f}"
    cv2.putText(canvas, stats_text, (graph_width - 200, 20), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1)
    return canvas

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.Canny(frame, 50, 150)
        inversed_edge = ~frame

        cv2.imshow('Canny Edge', frame)

        tm.stop()
        real_fps = tm.getFPS()
        fps_history.append(real_fps)

        # FPS 그래프 창 업데이트
        graph_img = render_fps_graph(list(fps_history))
        cv2.imshow('FPS Graph', graph_img)
        tm.reset(); tm.start()
        if cv2.waitKey(1) & 0xFF == 27:
            break

else: print("Camera not opened")

cap.release()
cv2.destroyAllWindows()