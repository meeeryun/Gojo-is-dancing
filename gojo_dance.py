import cv2 as cv
import numpy as np
import imageio
import os
import shutil

# Setting about video, gif->png, board's pattern and cellsize
chessboard_video_name = "chessboard1.mp4"
gif_path = "gojo_dance.gif"
png_dir = "./pngs"
board_pattern = (8,6)
cellsize = 25.0

# Character's Start position
start_x = 2
start_y = 2

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
gif_frame_delay = 3

# Alpha filter, because this is 
prev_rvec, prev_tvec = None, None
alpha = 0.6  # 0.5~0.8 권장. 낮을수록 부드럽지만 반응이 느려짐

# Video's K Matrix and Distortion values
K = np.array([ 
    [892.36877456, 0.0, 961.8528907],
    [0.0, 894.46553008, 541.85538616],
    [0.0, 0.0, 1.0]
])
dist_coeffs = np.array([-0.05734935, 0.23092948, 0.00236224, 0.00213651, -0.38708988])

# Utility function
def gif_to_png_sequence(gif_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gif = imageio.mimread(gif_path)
    for i, frame in enumerate(gif):
        frame = np.array(frame)
        if frame.shape[2] == 3:
            alpha_ch = np.ones((frame.shape[0], frame.shape[1], 1), dtype=np.uint8) * 255
            frame = np.concatenate((frame, alpha_ch), axis=2)
        if i == 0:
            mask = np.all(frame[:, :, :3] == [0, 0, 0], axis=2)
            frame[mask, 3] = 0
        imageio.imwrite(os.path.join(output_dir, f"frame_{i:03d}.png"), frame)

def load_png_sequence(folder):
    filenames = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    frames = []
    for name in filenames:
        img = cv.imread(os.path.join(folder, name), cv.IMREAD_UNCHANGED)
        if img is not None:
            frames.append(img)
    return frames

# Ready to start
if os.path.exists(png_dir):
    shutil.rmtree(png_dir)
gif_to_png_sequence(gif_path, png_dir)
png_frames = load_png_sequence(png_dir)

cap = cv.VideoCapture(chessboard_video_name)
if not cap.isOpened():
    print("Cannot open the video.")
    exit()

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

# I use 'mp4v' to increase video's Resolution
out = cv.VideoWriter('output_high_res.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

objp = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
objp[:, :2] = np.indices(board_pattern).T.reshape(-1, 2)
objp *= cellsize

frame_idx = 0
fast_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
scale = 0.5

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    small_frame = cv.resize(frame, (0, 0), fx=scale, fy=scale)
    gray_small = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray_small, board_pattern, fast_flags)

    if found:
        corners = corners / scale
        gray_full = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners = cv.cornerSubPix(gray_full, corners, (11, 11), (-1, -1), criteria)
        
        success, rvec, tvec = cv.solvePnP(objp, corners, K, dist_coeffs)

        if success:
            # Using alpha filter
            if prev_rvec is not None:
                rvec = rvec * alpha + prev_rvec * (1.0 - alpha)
                tvec = tvec * alpha + prev_tvec * (1.0 - alpha)
            prev_rvec, prev_tvec = rvec, tvec

            # Post the AR using my gif->png
            gif_index = (frame_idx // gif_frame_delay) % len(png_frames)
            overlay_frame = png_frames[gif_index]
            gif_h, gif_w = overlay_frame.shape[:2]

            aspect_ratio = gif_w / gif_h
            gif_w_mm = 2 * cellsize 
            gif_h_mm = gif_w_mm / aspect_ratio

            model_pts = np.float32([
                [0, 0, -gif_h_mm], [0, 0, 0], 
                [gif_w_mm, 0, -gif_h_mm], [gif_w_mm, 0, 0]
            ]) + np.float32([start_x * cellsize, start_y * cellsize, 0])

            img_pts, _ = cv.projectPoints(model_pts, rvec, tvec, K, dist_coeffs)
            dst_pts = img_pts.reshape(-1, 2).astype(np.float32)
            src_pts = np.float32([[0, 0], [0, gif_h], [gif_w, 0], [gif_w, gif_h]])

            M = cv.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv.warpPerspective(overlay_frame, M, (frame_width, frame_height),
                                        borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

            mask = (warped[:, :, 3] > 0).astype(np.uint8) * 255
            cv.copyTo(src=warped[:, :, :3], dst=frame, mask=mask)

    out.write(frame)
    frame_idx += 1

    # To upgrade resolution
    if frame_idx % 2 == 0:
        display_frame = cv.resize(frame, (1280, 720)) 
        cv.imshow('High-Res AR Processing', display_frame)

    if cv.waitKey(1) == 27: break

cap.release()
out.release()
cv.destroyAllWindows()
print("Complete. Check the 'output_high_res.mp4' in your file.")