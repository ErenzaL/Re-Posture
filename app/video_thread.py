# app/video_thread.py
import time
import threading

import cv2

from app.config import CONF
from app.pose_estimator import PoseEstimator


# ===========================================================
# [3] 비디오 스레드
# ===========================================================
class VideoThread(threading.Thread):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.running = True
        self.estimator = PoseEstimator(CONF["MODEL_FILE"])

    def run(self):
        cap = cv2.VideoCapture(CONF["CAM_IDX"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, CONF["FPS"])

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(CONF["FRAME_DELAY"])
                continue

            frame = cv2.flip(frame, 1)
            raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.app.data_latest_raw = raw_rgb

            if self.app.state_monitoring:
                processed_frame, state, score = self.estimator.process_frame(frame)
                final_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                if state:
                    self.app.data_status = "Good" if state == "good" else "Bad"
                    self.app.data_score = score
                    self.app.data_capture_frame = final_rgb
                else:
                    self.app.data_status = "None"
                    self.app.data_capture_frame = raw_rgb

                self.app.data_display_frame = final_rgb
            else:
                self.app.data_display_frame = raw_rgb
                self.app.data_status = "Ready"

            time.sleep(CONF["FRAME_DELAY"])

        cap.release()

    def stop(self):
        self.running = False
