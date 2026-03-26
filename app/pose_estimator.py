import os
import shutil
import tempfile

import cv2
import mediapipe as mp
import numpy as np

from app.config import CONF

# [0] 라이브러리 로드 (Keras)
try:
    from keras.models import load_model
except ImportError:
    load_model = None
    print("[ERROR] Keras/TensorFlow가 설치되지 않았습니다.")


# ===========================================================
# [2] AI 로직 (PoseEstimator) - [수정됨: 정확도 개선]
# ===========================================================
class PoseEstimator:
    def __init__(self, model_path):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils  # [추가] 그리기 유틸

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.model = self._load_model_safely(model_path)
        self.target_indices = set(range(0, 13))
        self.draw_color = (0, 255, 0)  # 사용자 화면 표시용

        # [핵심 수정 1] 모델이 학습한 방식(fristfile.py)과 동일한 색상/두께 정의
        # 점: (0, 0, 255) Red / 선: (0, 255, 0) Green
        self.skel_pt = self.mp_draw.DrawingSpec(
            color=(0, 0, 255), thickness=2, circle_radius=2
        )
        self.skel_line = self.mp_draw.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2
        )

    def _load_model_safely(self, filename):
        if load_model is None:
            return None
        if not os.path.exists(filename):
            print(f"[WARN] 모델 파일을 찾을 수 없습니다: {filename}")
            return None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
            tmp.close()
            shutil.copyfile(filename, tmp.name)
            m = load_model(tmp.name)
            os.remove(tmp.name)
            return m
        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {e}")
            return None

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        # MediaPipe는 RGB 입력을 받음
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if not res.pose_landmarks:
            return frame, "None", 0.0

        # 1. 사용자에게 보여줄 오버레이 (초록색 선, 단순화)
        overlay = frame.copy()
        landmarks = res.pose_landmarks.landmark

        for connection in self.mp_pose.POSE_CONNECTIONS:
            s, e = connection
            if s in self.target_indices and e in self.target_indices:
                x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
                x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
                cv2.line(overlay, (x1, y1), (x2, y2), self.draw_color, 2)

        for idx in self.target_indices:
            cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(overlay, (cx, cy), 5, self.draw_color, -1)

        # [핵심 수정 2] 모델 입력용 스켈레톤 이미지 생성 (fristfile.py 방식)
        # 흰색 배경 생성
        input_skel = np.full((h, w, 3), 255, np.uint8)

        # 지정된 색상(skel_pt, skel_line)으로 그리기 -> 이게 없으면 인식률 급락함
        self.mp_draw.draw_landmarks(
            input_skel,
            res.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.skel_pt,
            connection_drawing_spec=self.skel_line,
        )

        state, score = self._predict(input_skel)

        return overlay, state, score

    def _predict(self, skel_img):
        if self.model is None:
            return "unknown", 0.0

        # 이미지 전처리 (224x224, float32, /255.0)
        img = cv2.resize(skel_img, CONF["IMG_SIZE"]).astype("float32") / 255.0
        img = np.expand_dims(img, 0)

        try:
            # [핵심 수정 3] 예측 로직을 fristfile.py와 동일하게 변경
            # 모델 출력의 [0][0] 값이 '나쁜 자세일 확률'이라고 가정
            pred = self.model.predict(img, verbose=0)
            prob_bad = float(pred[0][0])

            prob_good = 1.0 - prob_bad

            # 0.5 이상이면 Bad, 아니면 Good
            if prob_bad >= 0.5:
                return "bad", (prob_good * 100)  # 혹은 (1.0-prob_good)*100 -> 즉 prob_bad*100
            else:
                return "good", (prob_good * 100)

        except Exception:
            return "unknown", 0.0
