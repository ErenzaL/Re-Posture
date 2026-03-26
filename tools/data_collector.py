import cv2
import time
import csv
import numpy as np
from pathlib import Path
import mediapipe as mp

# ================== CONFIG ==================
DATA_DIR = Path("data")
LABELS = ["good","bad"]
WEBCAM_INDEX = 0
BG_WHITE = True
MODEL_COMPLEXITY = 1
META_DIR = DATA_DIR / "meta"
CSV_PATH = META_DIR / "captures.csv"
# ============================================

current_label = LABELS[0]

def make_dirs(label):
    for sub in ["raw","skeleton","combined"]:
        (DATA_DIR / label / sub).mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

def blank_canvas(h, w):
    return np.full((h,w,3),255,np.uint8) if BG_WHITE else np.zeros((h,w,3),np.uint8)

def write_header_if_needed(csv_path: Path):
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "capture_id","timestamp_ms","label",
                "raw_path","skeleton_path","combined_path",
                "width","height",
                "shoulder_slope_deg","trunk_lean_deg","cva_deg","shoulder_span_rel"
            ])

def append_row(csv_path: Path, row: list):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# ---- Angle helpers (pixels → degrees)
def _angle_between(v, ref):
    v = v / (np.linalg.norm(v) + 1e-6)
    ref = ref / (np.linalg.norm(ref) + 1e-6)
    return np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(ref[1], ref[0]))

def compute_angles(frame_bgr, lm):
    """Returns (shoulder_slope_deg, trunk_lean_deg, cva_deg, shoulder_span_rel)."""
    h, w = frame_bgr.shape[:2]
    def xy(i):
        p = lm[i]; return np.array([p.x*w, p.y*h], dtype=np.float32)

    NOSE=0; L_SH=11; R_SH=12; L_HIP=23; R_HIP=24
    lsh, rsh = xy(L_SH), xy(R_SH)
    lip, rip = xy(L_HIP), xy(R_HIP)
    nose = xy(NOSE)
    sh_mid = (lsh + rsh)/2.0
    hip_mid = (lip + rip)/2.0

    shoulder_vec = rsh - lsh
    trunk_vec    = sh_mid - hip_mid

    shoulder_slope = _angle_between(shoulder_vec, np.array([1,0]))      # vs horizontal
    trunk_lean     = _angle_between(trunk_vec,   np.array([0,-1]))      # vs vertical (up)
    cva            = 90 - abs(_angle_between(nose - sh_mid, np.array([1,0])))  # forward-head proxy
    shoulder_span_rel = np.linalg.norm(rsh - lsh) / max(h,1)            # distance proxy

    return float(shoulder_slope), float(trunk_lean), float(cva), float(shoulder_span_rel)

# ---- MediaPipe
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=MODEL_COMPLEXITY)
skel_pt   = mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
skel_line = mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)

# ---- Camera
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    raise SystemExit

print("Controls:")
print("  1,2  = change posture label")
print("  s    = save (raw + skeleton + combined) + append angles to CSV")
print("  q    = quit")

make_dirs(current_label)
write_header_if_needed(CSV_PATH)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Mirror like selfie for consistency
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    # Live preview
    preview = frame.copy()
    if res.pose_landmarks:
        mp_draw.draw_landmarks(
            preview, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
            connection_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=2)
        )

    # HUD
    cv2.putText(preview, f"Label: {current_label}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Posture Collector (press 's' to save)", preview)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

    elif k in [ord(str(i)) for i in range(1,7)]:
        idx = int(chr(k)) - 1
        if 0 <= idx < len(LABELS):
            current_label = LABELS[idx]
            make_dirs(current_label)
            print(f"📌 Label → {current_label}")

    elif k == ord('s'):
        if not res.pose_landmarks:
            print("⚠️ No pose detected; not saving.")
            continue

        ts = int(time.time() * 1000)
        capture_id = f"{current_label}_{ts}"
        h, w = frame.shape[:2]

        # 1) RAW
        raw_path = DATA_DIR/current_label/"raw"/f"{capture_id}_raw.png"
        cv2.imwrite(str(raw_path), frame)

        # 2) SKELETON-ONLY (white/black bg)
        skeleton = blank_canvas(h, w)
        mp_draw.draw_landmarks(
            skeleton, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=skel_pt, connection_drawing_spec=skel_line
        )
        skel_path = DATA_DIR/current_label/"skeleton"/f"{capture_id}_skeleton.png"
        cv2.imwrite(str(skel_path), skeleton)

        # 3) COMBINED (overlay skeleton on raw)
        combined = frame.copy()
        mp_draw.draw_landmarks(
            combined, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=skel_pt, connection_drawing_spec=skel_line
        )
        comb_path = DATA_DIR/current_label/"combined"/f"{capture_id}_combined.png"
        cv2.imwrite(str(comb_path), combined)

        # 4) ANGLES → CSV
        ss, tl, cva, span = compute_angles(frame, res.pose_landmarks.landmark)
        append_row(CSV_PATH, [
            capture_id, ts, current_label,
            str(raw_path.as_posix()), str(skel_path.as_posix()), str(comb_path.as_posix()),
            w, h, round(ss,3), round(tl,3), round(cva,3), round(span,6)
        ])

        print(f"💾 Saved trio + CSV row: {capture_id}")

cap.release()
cv2.destroyAllWindows()
pose.close()