import sys
import tkinter as tk

# 실행 위치에 따라 둘 다 지원
try:
    # 프로젝트 루트에서: python -m app.main
    from app.ui import PostureApp
except ModuleNotFoundError:
    # app 폴더 안에서 직접: python main.py
    from ui import PostureApp


def main():
    root = tk.Tk()
    app = PostureApp(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
