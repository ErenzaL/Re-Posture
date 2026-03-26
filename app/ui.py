# app/ui.py
import sys
import cv2
import time
import os
import numpy as np
from datetime import datetime

import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk

# [0] 라이브러리 로드
try:
    from plyer import notification
except ImportError:
    notification = None

# 내부 모듈
from app.config import CONF
from app.video_thread import VideoThread


# ===========================================================
# [4] 메인 UI 클래스
# ===========================================================
class PostureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Posture Partner")
        self.root.geometry("1000x600")
        self.root.configure(bg=CONF["COLOR_BG"])

        # 데이터
        self.data_latest_raw = None
        self.data_display_frame = None
        self.data_capture_frame = None
        self.data_status = "Ready"
        self.data_score = 0.0

        # 상태
        self.state_monitoring = False
        self.state_mini_mode = False
        self.state_view_mode = "LIVE"

        # 통계/설정
        self.time_start = None
        self.time_stopped = 0.0
        self.time_good = 0.0
        self.time_bad = 0.0
        self.last_update_time = None

        self.time_bad_start = None
        self.time_last_snap = None

        self.cfg_warn_sec = CONF["DEFAULT_WARN_SEC"]
        self.cfg_save_sec = CONF["DEFAULT_SAVE_SEC"]
        self.cfg_mini_w = CONF["DEFAULT_MINI_W"]
        self.cfg_always_top = CONF["DEFAULT_ALWAYS_TOP"]

        # UI 리소스
        self.save_dir = self._create_save_dir()
        self.img_initial = self._create_placeholder_image("CAMERA OFF", color=(0, 0, 0))
        self.img_frozen = None
        self.img_review = None

        # 이미지 참조 보관용
        self.main_img_ref = None
        self.pip_img_ref = None
        self.mini_img_ref = None

        # 캔버스 ID
        self.id_main_img = None
        self.id_pip_img = None
        self.id_mini_img = None
        self.id_mini_text = None

        # 위젯 참조
        self.txt_log = None
        self.cvs_main = None
        self.frame_pip = None
        self.cvs_pip = None
        self.btn_setting = None
        self.win_mini = None
        self.cvs_mini = None

        # 바인딩
        self.root.bind("s", lambda e: self.take_snapshot())
        self.root.bind("w", lambda e: self.trigger_warning())

        self.build_ui()
        self.thread = VideoThread(self)
        self.thread.daemon = True
        self.thread.start()
        self.root.after(100, self.loop)

    def log(self, msg):
        print(f"[LOG] {msg}")
        if self.txt_log:
            try:
                self.txt_log.config(state="normal")
                self.txt_log.insert(
                    tk.END, f"[{datetime.now().strftime('%H:%M')}] {msg}\n"
                )
                self.txt_log.see(tk.END)
                self.txt_log.config(state="disabled")
            except:
                pass

    def _create_save_dir(self):
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        path = os.path.join(CONF["SAVE_DIR"], now)
        os.makedirs(path, exist_ok=True)
        return path

    def _create_placeholder_image(self, text, color=(0, 0, 0), base_img=None):
        if base_img is not None:
            img = base_img
        else:
            img = np.full((480, 640, 3), color, dtype=np.uint8)
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 1.2, 3)
        cx, cy = (w - tw) // 2, (h + th) // 2
        cv2.putText(img, text, (cx + 2, cy + 2), font, 1.2, (50, 50, 50), 5)
        cv2.putText(img, text, (cx, cy), font, 1.2, (255, 255, 255), 3)
        return img

    # -------------------------------------------------------
    # 1. UI Builder
    # -------------------------------------------------------
    def build_ui(self):
        for w in self.root.winfo_children():
            if isinstance(w, tk.Toplevel):
                continue
            w.destroy()

        self.id_main_img = None
        self.id_pip_img = None
        self.id_mini_img = None
        self.id_mini_text = None

        self.root.update_idletasks()

        if self.state_mini_mode:
            self._build_mini_layout()
        else:
            self._build_normal_layout()

    def _build_normal_layout(self):
        self.root.attributes("-topmost", False)
        self.root.overrideredirect(False)
        w, h = 1000, 600
        self.root.geometry(f"{w}x{h}")

        left_w = int(w * 0.7)
        right_w = w - left_w

        # [Left]
        left = tk.Frame(self.root, bg=CONF["COLOR_BG"])
        left.place(x=0, y=0, width=left_w, height=h)

        tk.Button(left, text="↘", command=self.toggle_mini_mode).place(
            x=55, y=10, width=35, height=30
        )
        tk.Label(left, text=os.path.basename(self.save_dir), bg=CONF["COLOR_BG"]).place(
            x=100, y=10
        )

        self.frame_video = tk.Frame(left, bg="black")
        self.frame_video.place(x=10, y=50, width=left_w - 20, height=h - 140)

        self.cvs_main = tk.Canvas(self.frame_video, bg="black", highlightthickness=0)
        self.cvs_main.pack(fill="both", expand=True)

        self.frame_pip = tk.Frame(
            self.frame_video,
            bg="black",
            highlightthickness=2,
            highlightbackground="yellow",
            cursor="hand2",
        )
        self.cvs_pip = tk.Canvas(self.frame_pip, bg="black", highlightthickness=0)
        self.cvs_pip.pack(fill="both", expand=True)
        self.frame_pip.place(relx=2.0, rely=2.0)

        btn_txt = "■ STOP" if self.state_monitoring else "▶ START"
        btn_bg = CONF["COLOR_BAD"] if self.state_monitoring else CONF["COLOR_ACCENT"]
        self.btn_action = tk.Button(
            left,
            text=btn_txt,
            bg=btn_bg,
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.toggle_monitoring,
        )
        self.btn_action.place(x=10, y=h - 80, width=left_w - 20, height=60)

        # [Right]
        right = tk.Frame(self.root, bg=CONF["COLOR_BG"])
        right.place(x=left_w, y=0, width=right_w, height=h)

        tk.Button(right, text="X", command=self.on_close, bg="#FFDDDD").place(
            x=right_w - 40, y=10, width=30, height=30
        )
        self.btn_setting = tk.Button(right, text="⚙", command=self.open_settings)
        self.btn_setting.place(x=right_w - 80, y=10, width=30, height=30)

        unit_h = (h - 60) / 12
        cur_y = 50

        self.lbl_status, self.lbl_conf = self._make_card(
            right, 10, cur_y, right_w - 20, unit_h * 3.0, "READY", "Conf: 0%"
        )
        cur_y += unit_h * 3.0 + 10

        self.lbl_time, self.lbl_sub_time = self._make_card(
            right, 10, cur_y, right_w - 20, unit_h * 3.0, "00:00:00", "G:00 B:00"
        )
        cur_y += unit_h * 3.0 + 10

        f_log = tk.Frame(right, bg=CONF["COLOR_CARD_BG"], bd=1, relief="solid")
        f_log.place(x=10, y=cur_y, width=right_w - 20, height=unit_h * 3.0)
        self.txt_log = scrolledtext.ScrolledText(
            f_log, bg="#FAFAFA", state="disabled"
        )
        self.txt_log.place(relx=0, rely=0, relwidth=1, relheight=1)
        cur_y += unit_h * 3.0 + 10

        rem_h = h - cur_y - 20

        # 갤러리 영역 스크롤바 유지
        f_gal = tk.Frame(right, bg=CONF["COLOR_CARD_BG"], bd=1, relief="solid")
        f_gal.place(x=10, y=cur_y, width=right_w - 20, height=max(50, rem_h))

        scroll_x = tk.Scrollbar(f_gal, orient="horizontal")

        cvs_gal = tk.Canvas(
            f_gal, bg=CONF["COLOR_CARD_BG"], xscrollcommand=scroll_x.set
        )
        scroll_x.config(command=cvs_gal.xview)

        scroll_x.pack(side="bottom", fill="x")
        cvs_gal.pack(side="top", fill="both", expand=True)

        self.frame_gal = tk.Frame(cvs_gal, bg=CONF["COLOR_CARD_BG"])
        cvs_gal.create_window((0, 0), window=self.frame_gal, anchor="nw")
        self.frame_gal.bind(
            "<Configure>", lambda e: cvs_gal.configure(scrollregion=cvs_gal.bbox("all"))
        )

        self.log("정상 모드 전환")

    def _build_mini_layout(self):
        self.root.withdraw()

        w = self.cfg_mini_w
        h = int(self.cfg_mini_w * 3 / 4)

        self.win_mini = tk.Toplevel(self.root)
        self.win_mini.geometry(f"{w}x{h}")
        self.win_mini.overrideredirect(False)
        self.win_mini.attributes("-topmost", self.cfg_always_top)
        self.win_mini.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cvs_mini = tk.Canvas(
            self.win_mini, bg="black", highlightthickness=0, width=w, height=h
        )
        self.cvs_mini.pack(fill="both", expand=True)
        self.cvs_mini.bind("<Double-Button-1>", lambda e: self.toggle_mini_mode())

        self.log("미니 모드 전환 (더블클릭 복귀)")

    def _make_card(self, parent, x, y, w, h, main_txt, sub_txt):
        f = tk.Frame(parent, bg=CONF["COLOR_CARD_BG"], bd=1, relief="solid")
        f.place(x=x, y=y, width=w, height=h)
        l1 = tk.Label(
            f,
            text=main_txt,
            font=("Arial", 20, "bold"),
            bg=CONF["COLOR_CARD_BG"],
        )
        l1.place(relx=0.5, rely=0.4, anchor="center")
        l2 = None
        if sub_txt:
            l2 = tk.Label(f, text=sub_txt, bg=CONF["COLOR_CARD_BG"])
            l2.place(relx=0.5, rely=0.75, anchor="center")
        return l1, l2

    # -------------------------------------------------------
    # 2. Logic & Loop
    # -------------------------------------------------------
    def loop(self):
        try:
            if self.state_monitoring and self.time_start:
                now = time.time()
                dt = now - self.last_update_time
                self.last_update_time = now

                if self.data_status == "Good":
                    self.time_good += dt
                    self.time_bad_start = None
                    if hasattr(self, "lbl_status"):
                        self.lbl_status.config(text="Good", fg=CONF["COLOR_GOOD"])
                elif self.data_status == "Bad":
                    self.time_bad += dt
                    if hasattr(self, "lbl_status"):
                        self.lbl_status.config(text="Bad", fg=CONF["COLOR_BAD"])

                    if self.time_bad_start is None:
                        self.time_bad_start = now
                    if now - self.time_bad_start >= self.cfg_warn_sec:
                        if (
                            self.time_last_snap is None
                            or now - self.time_last_snap >= self.cfg_save_sec
                        ):
                            self.trigger_warning()
                            self.time_last_snap = now

                if hasattr(self, "lbl_time"):
                    total = now - self.time_start + self.time_stopped
                    self.lbl_time.config(text=self._format_time(total))
                    self.lbl_sub_time.config(
                        text=f"G: {self._format_time(self.time_good)}  B: {self._format_time(self.time_bad)}"
                    )
                    if hasattr(self, "lbl_conf"):
                        self.lbl_conf.config(text=f"Conf: {self.data_score:.1f}%")

            self._draw_video()

        except Exception:
            pass
        self.root.after(30, self.loop)

    def _draw_video(self):
        target = None
        if self.state_view_mode == "REVIEW" and self.img_review is not None:
            target = self.img_review
        elif self.state_monitoring:
            target = self.data_display_frame
        else:
            target = self.img_frozen if self.img_frozen is not None else self.img_initial

        if target is None:
            return

        if self.state_mini_mode:
            self._draw_mini_frame(target)
        else:
            self._draw_main_frame(target)

    def _draw_mini_frame(self, img):
        if not self.cvs_mini:
            return
        try:
            fixed_w = self.cfg_mini_w
            fixed_h = int(fixed_w * 0.75)

            resized = cv2.resize(img, (fixed_w, fixed_h))
            tk_img = ImageTk.PhotoImage(image=Image.fromarray(resized))
            self.mini_img_ref = tk_img

            if self.id_mini_img is None:
                self.id_mini_img = self.cvs_mini.create_image(
                    0, 0, anchor="nw", image=tk_img
                )
            else:
                try:
                    self.cvs_mini.itemconfig(self.id_mini_img, image=tk_img)
                except:
                    self.id_mini_img = self.cvs_mini.create_image(
                        0, 0, anchor="nw", image=tk_img
                    )

            if self.state_monitoring:
                txt = f"{self.data_status} ({self.data_score:.1f}%)"
                col = "#00FF00" if self.data_status == "Good" else "#FF0000"
                if self.id_mini_text is None:
                    self.id_mini_text = self.cvs_mini.create_text(
                        10,
                        20,
                        text=txt,
                        fill=col,
                        font=("Arial", 16, "bold"),
                        anchor="nw",
                    )
                else:
                    try:
                        self.cvs_mini.itemconfig(self.id_mini_text, text=txt, fill=col)
                        self.cvs_mini.tag_raise(self.id_mini_text)
                    except:
                        self.id_mini_text = self.cvs_mini.create_text(
                            10,
                            20,
                            text=txt,
                            fill=col,
                            font=("Arial", 16, "bold"),
                            anchor="nw",
                        )
        except:
            pass

    def _draw_main_frame(self, img):
        if not self.cvs_main:
            return
        self._paint_canvas(self.cvs_main, img, "main")

        if self.state_view_mode == "REVIEW" and self.cvs_pip:
            src = (
                self.data_display_frame
                if self.state_monitoring
                else self.data_latest_raw
            )
            if src is not None:
                self._paint_canvas(self.cvs_pip, src, "pip")

    def _paint_canvas(self, canvas, img, tag):
        try:
            cw, ch = canvas.winfo_width(), canvas.winfo_height()
            if cw < 5:
                return

            h, w = img.shape[:2]
            scale = max(cw / w, ch / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (nw, nh))
            tk_img = ImageTk.PhotoImage(image=Image.fromarray(resized))

            cx, cy = cw // 2, ch // 2

            cur_id = self.id_main_img if tag == "main" else self.id_pip_img

            if cur_id is None:
                new_id = canvas.create_image(cx, cy, anchor="center", image=tk_img)
                if tag == "main":
                    self.id_main_img = new_id
                else:
                    self.id_pip_img = new_id
            else:
                try:
                    canvas.coords(cur_id, cx, cy)
                    canvas.itemconfig(cur_id, image=tk_img)
                except:
                    new_id = canvas.create_image(cx, cy, anchor="center", image=tk_img)
                    if tag == "main":
                        self.id_main_img = new_id
                    else:
                        self.id_pip_img = new_id

            canvas.image = tk_img
        except:
            pass

    # -------------------------------------------------------
    # Actions
    # -------------------------------------------------------
    def toggle_monitoring(self):
        if not self.state_monitoring:
            self.state_monitoring = True
            self.time_start = time.time()
            self.last_update_time = self.time_start
            if hasattr(self, "btn_action"):
                self.btn_action.config(text="■ STOP", bg=CONF["COLOR_BAD"])
            self.img_frozen = None
            self.log("모니터링 시작")
        else:
            self.state_monitoring = False
            if self.time_start:
                self.time_stopped += time.time() - self.time_start
            self.time_start = None
            if hasattr(self, "btn_action"):
                self.btn_action.config(text="▶ START", bg=CONF["COLOR_ACCENT"])

            target = (
                self.data_display_frame
                if self.data_display_frame is not None
                else self.data_latest_raw
            )
            if target is not None:
                self.img_frozen = self._process_frozen_image(target)
            if hasattr(self, "lbl_status"):
                self.lbl_status.config(text="PAUSED", fg="gray")
            self.log("모니터링 종료")

    def toggle_mini_mode(self):
        if not self.state_mini_mode:
            self.state_mini_mode = True
            self.root.withdraw()

            w = self.cfg_mini_w
            h = int(w * 0.75)
            self.win_mini = tk.Toplevel(self.root)
            self.win_mini.geometry(f"{w}x{h}")
            self.win_mini.overrideredirect(False)
            self.win_mini.attributes("-topmost", self.cfg_always_top)
            self.win_mini.protocol("WM_DELETE_WINDOW", self.on_close)

            self.cvs_mini = tk.Canvas(
                self.win_mini, bg="black", highlightthickness=0, cursor="hand2"
            )
            self.cvs_mini.pack(fill="both", expand=True)
            self.cvs_mini.bind(
                "<Double-Button-1>", lambda e: self.toggle_mini_mode()
            )

            self.id_mini_img = None
            self.id_mini_text = None
            self.log("미니 모드 전환 (더블클릭 복귀)")
        else:
            self.state_mini_mode = False
            if self.win_mini:
                self.win_mini.destroy()
                self.win_mini = None

            self.root.deiconify()
            self.id_main_img = None
            self.log("정상 모드 복귀")

    def enable_review(self, fpath):
        if not os.path.exists(fpath):
            return
        bgr = cv2.imread(fpath)
        if bgr is None:
            return
        self.img_review = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.state_view_mode = "REVIEW"
        self.log("리뷰 모드 진입")

        if self.frame_pip:
            self.frame_pip.lift()
            self.frame_pip.place(
                relx=1.0, rely=0.0, anchor="ne", x=-10, y=10, width=200, height=150
            )
            self.frame_pip.bind("<Button-1>", self.disable_review, add="+")
            self.cvs_pip.bind("<Button-1>", self.disable_review, add="+")

    def disable_review(self, event=None):
        self.state_view_mode = "LIVE"
        self.img_review = None
        if self.frame_pip:
            self.frame_pip.unbind("<Button-1>")
            self.cvs_pip.unbind("<Button-1>")
            self.frame_pip.place(relx=2.0, rely=2.0)
        self.log("라이브 복귀")

    def take_snapshot(self):
        if self.data_capture_frame is None:
            return
        ts = datetime.now().strftime("%H%M%S")
        fpath = os.path.join(self.save_dir, f"cap_{ts}.jpg")
        cv2.imwrite(fpath, cv2.cvtColor(self.data_capture_frame, cv2.COLOR_RGB2BGR))
        self._add_gallery_thumb(fpath, self.data_capture_frame)
        self.log("스냅샷 저장됨")

    def trigger_warning(self):
        self.take_snapshot()
        try:
            if notification:
                notification.notify(title="자세 경고", message="허리를 펴세요!", timeout=3)
        except:
            pass
        self.log("!!! 경고 발생 !!!")

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("설정")
        win.geometry("300x300")
        if self.btn_setting:
            try:
                bx = self.btn_setting.winfo_rootx()
                by = self.btn_setting.winfo_rooty()
                win.geometry(f"300x300+{bx-310}+{by}")
            except:
                pass
        win.transient(self.root)
        win.grab_set()

        val_warn = tk.IntVar(value=self.cfg_warn_sec // 60)
        val_save = tk.IntVar(value=self.cfg_save_sec // 60)
        val_width = tk.IntVar(value=self.cfg_mini_w)
        val_top = tk.BooleanVar(value=self.cfg_always_top)

        tk.Label(win, text="경고 알림 대기 (분)").pack(pady=5)
        tk.Scale(
            win, from_=1, to=60, orient="horizontal", variable=val_warn, length=200
        ).pack()

        tk.Label(win, text="사진 저장 간격 (분)").pack(pady=5)
        tk.Scale(
            win, from_=1, to=60, orient="horizontal", variable=val_save, length=200
        ).pack()

        tk.Label(win, text="미니모드 너비 (px)").pack(pady=5)
        tk.Entry(win, textvariable=val_width).pack()

        tk.Checkbutton(
            win, text="미니모드 항상 위에 고정", variable=val_top
        ).pack(pady=10)

        def save():
            self.cfg_warn_sec = val_warn.get() * 60
            self.cfg_save_sec = val_save.get() * 60
            self.cfg_mini_w = val_width.get()
            self.cfg_always_top = val_top.get()
            self.log("설정 저장됨")
            win.destroy()

        tk.Button(win, text="저장", command=save, bg=CONF["COLOR_BTN_BG"]).pack(
            pady=20
        )

    def _process_frozen_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        darkened = cv2.addWeighted(
            bgr_gray, 0.6, np.zeros(bgr_gray.shape, bgr_gray.dtype), 0, 0
        )
        return self._create_placeholder_image(
            "CAMERA OFF", color=None, base_img=darkened
        )

    def _add_gallery_thumb(self, fpath, rgb):
        if not hasattr(self, "frame_gal") or not self.frame_gal.winfo_exists():
            return
        h, w = rgb.shape[:2]
        ratio = 40 / h
        thumb = cv2.resize(rgb, (int(w * ratio), 40))
        img = ImageTk.PhotoImage(Image.fromarray(thumb))
        btn = tk.Button(
            self.frame_gal, image=img, command=lambda: self.enable_review(fpath)
        )
        btn.image = img
        btn.pack(side="left", padx=2, pady=2)

    def _format_time(self, sec):
        sec = int(sec)
        return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"

    def on_close(self):
        print("[SYSTEM] 종료")
        self.thread.stop()
        self.root.destroy()
        sys.exit()
