import cv2
import numpy as np
import mediapipe as mp
import os

def load_glasses(path):
    """Tải ảnh kính và đảm bảo ảnh có kênh alpha, xử lý ảnh có nền phức tạp"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file {path}")
    
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"File {path} không phải ảnh hợp lệ")
    
    # Nếu ảnh không có kênh alpha (chỉ có BGR), ta sẽ tạo alpha
    if img.shape[-1] == 3:  # Ảnh không có kênh alpha
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)  # Làm mượt alpha
        img = cv2.merge((img, alpha))  # Thêm kênh alpha vào ảnh
    
    return img

def remove_background(image):
    """Dùng GrabCut để tách nền của ảnh kính"""
    # Kiểm tra nếu ảnh có 4 kênh (BGR + Alpha)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Khởi tạo mask và các mô hình cho GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Cắt bỏ phần nền bằng GrabCut (thử dùng phương pháp tự động hơn)
    rect = (10, 10, image.shape[1]-10, image.shape[0]-10)  # Vùng bao quanh kính
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Đánh dấu các khu vực nền là 0 và đối tượng là 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask2[:, :, np.newaxis]

    return image

# Khởi tạo thư viện Mediapipe cho nhận diện khuôn mặt
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Load ảnh kính
glasses = load_glasses('glasses7.jpeg')

# Các điểm chuẩn trên khuôn mặt để căn chỉnh kính
LEFT_EYE = 33  # Khóe mắt trái
RIGHT_EYE = 263  # Khóe mắt phải
NOSE = 6  # Mũi làm tham chiếu chiều cao

cap = cv2.VideoCapture(0)


def process_frame(frame: np.ndarray, glasses: np.ndarray) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        def get_coord(idx):
            landmark = face_landmarks.landmark[idx]
            return int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

        try:
            left_eye = get_coord(LEFT_EYE)
            right_eye = get_coord(RIGHT_EYE)
            nose = get_coord(NOSE)

            eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            eye_width = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            angle = -np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
            scale = eye_width * 2.0 / glasses.shape[1]

            resized_glasses = cv2.resize(glasses, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            glass_img = resized_glasses[:, :, :3]
            glass_alpha = resized_glasses[:, :, 3] / 255.0

            pad = 50
            glass_img = cv2.copyMakeBorder(glass_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            glass_alpha = cv2.copyMakeBorder(glass_alpha, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

            M = cv2.getRotationMatrix2D((glass_img.shape[1] // 2, glass_img.shape[0] // 2), angle, 1)
            rotated_img = cv2.warpAffine(glass_img, M, (glass_img.shape[1], glass_img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            rotated_alpha = cv2.warpAffine(glass_alpha, M, (glass_alpha.shape[1], glass_alpha.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            x = eye_center[0] - rotated_img.shape[1] // 2
            y = eye_center[1] - rotated_img.shape[0] // 2

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + rotated_img.shape[1]), min(frame.shape[0], y + rotated_img.shape[0])

            if x2 > x1 and y2 > y1:
                region = frame[y1:y2, x1:x2]
                glass_part = rotated_img[y1 - y:y2 - y, x1 - x:x2 - x]
                alpha_part = rotated_alpha[y1 - y:y2 - y, x1 - x:x2 - x]
                inv_alpha = 1.0 - alpha_part

                for c in range(3):
                    region[:, :, c] = (glass_part[:, :, c] * alpha_part) + (region[:, :, c] * inv_alpha)

        except Exception as e:
            print(f"Lỗi xử lý: {str(e)}")

    return frame



