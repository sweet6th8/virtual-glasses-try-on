import cv2
import numpy as np
import mediapipe as mp
import os

def load_glasses(path):
    """Tải ảnh kính và xử lý alpha channel, tự động chuyển ảnh lỗi về đúng định dạng"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Không tải được ảnh kính từ {path}")
    # Nếu ảnh chỉ có 1 kênh (grayscale), chuyển sang BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Nếu ảnh chỉ có 3 kênh, thêm alpha channel
    if img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        alpha = cv2.GaussianBlur(alpha, (7,7), 0)
        img = cv2.merge([img[:, :, 0], img[:, :, 1], img[:, :, 2], alpha])
    # Nếu ảnh đã có 4 kênh thì giữ nguyên
    return img

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,  # Giảm confidence để dễ detect hơn
    min_tracking_confidence=0.5
)
glasses = load_glasses('img/glasses5.png')
# Sử dụng các landmark chính xác cho mắt
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

def get_eye_center(landmarks, eye_indices):
    """Tính tọa độ trung tâm mắt từ các landmark"""
    points = [
        (landmarks.landmark[i].x, landmarks.landmark[i].y)
        for i in eye_indices
    ]
    return np.mean(points, axis=0)

def process_frame(frame: np.ndarray, glasses: np.ndarray) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        print("Không detect được khuôn mặt!")
        return frame
    face_landmarks = results.multi_face_landmarks[0]
    try:
        # Lấy tọa độ normalized, chuyển sang pixel
        left_eye_norm = get_eye_center(face_landmarks, LEFT_EYE_INDICES)
        right_eye_norm = get_eye_center(face_landmarks, RIGHT_EYE_INDICES)
        h, w = frame.shape[:2]
        left_eye = (left_eye_norm[0] * w, left_eye_norm[1] * h)
        right_eye = (right_eye_norm[0] * w, right_eye_norm[1] * h)
        eye_center = ((left_eye[0] + right_eye[0])/2, (left_eye[1] + right_eye[1])/2)
        eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
        print(f"eye_distance: {eye_distance}")
        if eye_distance < 5:  # Giá trị này tùy chỉnh, tránh lỗi chia cho số nhỏ
            print("eye_distance quá nhỏ, bỏ qua frame này")
            return frame
        scale_factor = eye_distance * 3.2 / glasses.shape[1]
        new_width = int(glasses.shape[1] * scale_factor)
        new_height = int(glasses.shape[0] * scale_factor)
        if new_width <= 0 or new_height <= 0:
            print("Kích thước kính <= 0, bỏ qua frame này")
            return frame
        angle = -np.degrees(np.arctan2(right_eye[1]-left_eye[1], right_eye[0]-left_eye[0]))
        resized_glasses = cv2.resize(glasses, (new_width, new_height))
        M = cv2.getRotationMatrix2D(
            (new_width//2, new_height//2),
            angle,
            1.0
        )
        rotated_glasses = cv2.warpAffine(
            resized_glasses,
            M,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,0,0,0)
        )
        # Lật kính ngang lại cho đúng chiều
        rotated_glasses = cv2.flip(rotated_glasses, 1)
        # Xoay kính 180 độ nếu bị ngược dọc
        rotated_glasses = cv2.rotate(rotated_glasses, cv2.ROTATE_180)
        x = int(eye_center[0] - new_width // 2)
        y = int(eye_center[1] - new_height // 2.2)
        overlay_img(frame, rotated_glasses, x, y)
    except Exception as e:
        print(f"Lỗi khi xử lý kính: {str(e)}")
    return frame

def overlay_img(background, overlay, x, y):
    """Overlay ảnh RGBA lên BGR, kiểm tra biên"""
    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]
    if x < 0:
        overlay = overlay[:, -x:]
        w = overlay.shape[1]
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        h = overlay.shape[0]
        y = 0
    if x + w > bg_w:
        overlay = overlay[:, :bg_w - x]
        w = overlay.shape[1]
    if y + h > bg_h:
        overlay = overlay[:bg_h - y, :]
        h = overlay.shape[0]
    if w <= 0 or h <= 0:
        return
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            overlay[:, :, c] * alpha + background[y:y+h, x:x+w, c] * (1 - alpha)
        )