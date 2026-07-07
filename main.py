import tkinter as tk
from PIL import Image, ImageTk
import cv2
from deepface import DeepFace
import requests
import threading
import time
import os

# ==============================
# 🔐 TELEGRAM CONFIG
# ==============================
TELEGRAM_TOKEN = "8838411941:AAEK-3T8lFWAfDUkVi4TOQl80qFW4o9E1vs"
CHAT_ID = 7214834118

# ==============================                
# GLOBALS
# ==============================
latest_caption = "No detection"
latest_frame = None
latest_boxes = []
latest_names = []

# ==============================
# TELEGRAM FUNCTIONS
# ==============================
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text})
    except:
        pass

# ==============================
# RECORD VIDEO (LIVE DATA)
# ==============================
def record_and_send_video():
    def task():
        global latest_frame, latest_boxes, latest_names, latest_caption

        filename = "alert.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (600, 400))

        start = time.time()

        while (time.time() - start) < 7:

            if latest_frame is None:
                continue

            frame = latest_frame.copy()

            # ✅ LIVE DRAWING
            for (box, info) in zip(latest_boxes, latest_names):
                (x, y, w, h) = box
                (name, accuracy) = info

                text = f"{name} ({accuracy}%)"
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            out.write(frame)
            time.sleep(0.03)

        out.release()
        time.sleep(1)

        # ✅ SEND VIDEO WITH SAME CAPTION
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo"
            with open(filename, "rb") as v:
                requests.post(
                    url,
                    data={"chat_id": CHAT_ID, "caption": latest_caption},
                    files={"video": v}
                )
        except Exception as e:
            print("Video Error:", e)

    threading.Thread(target=task).start()

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==============================
# GUI
# ==============================
root = tk.Tk()
root.title("AI Surveillance Dashboard")
root.geometry("1000x650")
root.configure(bg="#0f172a")

title = tk.Label(root, text="🔍 AI Surveillance Dashboard",
                 font=("Segoe UI", 24, "bold"),
                 fg="#00ffc3", bg="#0f172a")
title.pack(pady=15)

video_frame = tk.Frame(root, bg="#1e293b", bd=3, relief="ridge")
video_frame.pack(pady=10)

video_label = tk.Label(video_frame)
video_label.pack()

status_label = tk.Label(root, text="Status: Waiting...",
                        font=("Segoe UI", 12),
                        fg="#22c55e", bg="#0f172a")
status_label.pack(pady=5)

name_label = tk.Label(root, text="Detected: None",
                      font=("Segoe UI", 16, "bold"),
                      fg="#38bdf8", bg="#0f172a")
name_label.pack(pady=5)

count_label = tk.Label(root, text="Faces Count: 0",
                       font=("Segoe UI", 14, "bold"),
                       fg="#facc15", bg="#0f172a")
count_label.pack(pady=5)

btn_frame = tk.Frame(root, bg="#0f172a")
btn_frame.pack(pady=15)

running = False
last_sent_name = None
video_sent = False

def start_camera():
    global running
    running = True
    update_frame()

def stop_camera():
    global running
    running = False
    status_label.config(text="Status: Stopped")

tk.Button(btn_frame, text="▶ Start", command=start_camera,
          bg="#00ffc3", fg="black",
          font=("Segoe UI", 12, "bold"), width=12).grid(row=0, column=0, padx=10)

tk.Button(btn_frame, text="⏹ Stop", command=stop_camera,
          bg="#ef4444", fg="white",
          font=("Segoe UI", 12, "bold"), width=12).grid(row=0, column=1, padx=10)

# ==============================
# MAIN LOOP
# ==============================
def update_frame():
    global last_sent_name, video_sent, latest_caption
    global latest_frame, latest_boxes, latest_names

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (600, 400))

    latest_frame = frame.copy()
    latest_boxes = []
    latest_names = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    total_faces = len(faces)
    caption_list = []

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        try:
            result = DeepFace.find(
                img_path=face_img,
                db_path="dataset",
                enforce_detection=False,
                model_name="Facenet512"
            )

            if len(result[0]) > 0:
                best = result[0].iloc[0]
                identity = best['identity']
                name = identity.split("\\")[-2]
                distance = best['distance']
                accuracy = int((1 - distance) * 100)

                if accuracy < 75:
                    name = "Unknown"
            else:
                name = "Unknown"
                accuracy = 0

        except:
            name = "Unknown"
            accuracy = 0

        latest_boxes.append((x, y, w, h))
        latest_names.append((name, accuracy))

        caption_list.append(f"{name} ({accuracy}%)")

        # DRAW ON SCREEN
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} ({accuracy}%)",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # TRIGGER
        if name != "Unknown" and accuracy >= 75:
            if name != last_sent_name:
                send_telegram_message(f"✅ {name} ({accuracy}%) | Faces: {total_faces}")
                last_sent_name = name

            if not video_sent:
                video_sent = True
                record_and_send_video()

    latest_caption = f"{', '.join(caption_list)} | Faces: {total_faces}"

    status_label.config(text="Status: Running")
    name_label.config(text=f"Detected: {', '.join(caption_list)}")
    count_label.config(text=f"Faces Count: {total_faces}")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# ==============================
# RUN
# ==============================
root.mainloop()
cap.release()
cv2.destroyAllWindows()