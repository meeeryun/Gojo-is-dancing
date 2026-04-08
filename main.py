import cv2 as cv
import numpy as np

# --- Video & Board 설정 ---
video = cv.VideoCapture("chessboard1.mp4")
board_pattern = (8,6)    # 체스보드 내부 코너 수 (columns, rows)
board_cellsize = 25       # mm 단위

# --- 카메라 캘리브레이션 값 (캘리브레이션 후 얻은 값으로 교체) ---
K = np.array([ 
    [892.36877456, 0.0, 961.8528907],
    [0.0, 894.46553008, 541.85538616],
    [0.0, 0.0, 1.0]
])
dist_coeff = np.array([-0.05734935, 0.23092948, 0.00236224, 0.00213651, -0.38708988])

board_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- 3D 객체 포인트 정의 ---
obj_points = board_cellsize * np.array([[c,r,0] 
                                        for r in range(board_pattern[1]) 
                                        for c in range(board_pattern[0])], dtype=np.float32)

# --- 박스 3D 좌표 (체스보드 좌표계 기준, Z축 양수: 위로) ---
box_lower = board_cellsize * np.array([[4,2,0], [5,2,0], [5,4,0], [4,4,0]], dtype=np.float32)
box_upper = board_cellsize * np.array([[4,2,-1], [5,2,-1], [5,4,-1], [4,4,-1]], dtype=np.float32)

while True:
    valid, img = video.read()
    if not valid:
        break

    complete, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    
    if complete:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)

        cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)
        cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2)

        for b, t in zip(line_lower, line_upper):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

            R, _ = cv.Rodrigues(rvec) 
            p = (-R.T @ tvec).flatten()
            info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
            cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0)) 

    # --- 결과 영상 출력 ---
    cv.imshow("Pose Estimation", img)
    if cv.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
        break

video.release()
cv.destroyAllWindows()