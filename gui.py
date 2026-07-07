import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os

from recognizer import recognize_face
from telegram_bot import send_message, send_video


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Face Recognition System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#0f172a")

        # ===== TITLE =====
        self.title = tk.Label(
            root,
            text="🔍 AI Surveillance Dashboard",
            font=("Segoe UI", 22, "bold"),
            fg="#00ffcc",
            bg="#0f172a"
        )
        self.title.pack(pady=10)

        # ===== VIDEO =====
        self.video_label = tk.Label(root, bd=2, relief="solid")
        self.video_label.pack()

        # ===== STATUS =====
        self.status = tk.Label(
            root,
            text="Status: Waiting...",
            fg="#22c55e",
            bg="#0f172a",
            font=("Segoe UI", 12)
        )
        self.status.pack(pady=10)

        # ===== DETECTED =====
        self.name_label = tk.Label(
            root,
            text="Detected: None",
            fg="#38bdf8",
            bg="#0f172a",
            font=("Segoe UI", 14, "bold")
        )
        self.name_label.pack()

        # ===== BUTTONS =====
        btn_frame = tk.Frame(root, bg="#0f172a")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Start", command=self.start_camera,
                  bg="#00ffcc", width=12).grid(row=0, column=0, padx=10)

        tk.Button(btn_frame, text="Stop", command=self.stop_camera,
                  bg="#ef4444", fg="white", width=12).grid(row=0, column=1, padx=10)

        # ===== CAMERA =====
        self.cap = None
        self.running = False

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.detected_names = set()

        # ===== PERFORMANCE =====
        self.frame_count = 0

        # ===== MEMORY =====
        self.face_memory = {}

        # ===== TRACKING =====
        self.tracked_faces = {}
        self.face_id_count = 0

        # ===== VIDEO RECORDING =====
        self.recording = False
        self.video_writer = None
        self.record_frames = 0

    # ===== TRACK FACE =====
    def get_face_id(self, x, y):
        for fid, (fx, fy) in self.tracked_faces.items():
            if abs(x - fx) < 60 and abs(y - fy) < 60:
                self.tracked_faces[fid] = (x, y)
                return fid

        self.face_id_count += 1
        self.tracked_faces[self.face_id_count] = (x, y)
        return self.face_id_count

    # ===== START CAMERA =====
    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()

    # ===== STOP CAMERA =====
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image="")
        self.status.config(text="Status: Stopped")

    # ===== START RECORDING =====
    def start_recording(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter("alert.avi", fourcc, 20.0, (480, 360))
        self.recording = True
        self.record_frames = 0
        print("🎥 Recording started")

    # ===== MAIN LOOP =====
    def update_frame(self):
        if not self.running:
            return

        self.frame_count += 1
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (480, 360))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5
            )

            names_in_frame = []

            for (x, y, w, h) in faces:

                if w < 60 or h < 60:
                    continue

                face_crop = frame[y:y+h, x:x+w]
                face_id = self.get_face_id(x, y)

                if face_id not in self.face_memory:
                    self.face_memory[face_id] = {"name": "Unknown", "acc": 0}

                # ===== RECOGNITION =====
                if self.frame_count % 10 == 0:
                    name, acc = recognize_face(face_crop)

                    if acc >= 70:
                        old_acc = self.face_memory[face_id]["acc"]
                        new_acc = int((old_acc + acc) / 2)

                        self.face_memory[face_id] = {
                            "name": name,
                            "acc": new_acc
                        }

                data = self.face_memory[face_id]

                display_name = (
                    f"{data['name']} ({data['acc']}%)"
                    if data["name"] != "Unknown"
                    else "Unknown"
                )

                names_in_frame.append(display_name)

                # ===== DRAW =====
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, display_name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)

                # ===== TELEGRAM ALERT =====
                if data["name"] != "Unknown" and data["name"] not in self.detected_names:

                    send_message(f"✅ Detected: {data['name']} ({data['acc']}%)")

                    # 🎥 START RECORDING
                    self.start_recording()

                    self.detected_names.add(data["name"])

            # ===== VIDEO RECORD =====
            if self.recording:
                self.video_writer.write(frame)
                self.record_frames += 1

                if self.record_frames > 160:  # ~8 sec
                    self.recording = False
                    self.video_writer.release()

                    print("📤 Sending video...")

                    send_video("alert.avi")

            # ===== UI UPDATE =====
            unique_names = list(set(names_in_frame))

            if unique_names and unique_names != ["Unknown"]:
                self.name_label.config(text=f"Detected: {', '.join(unique_names)}")
                self.status.config(text="Status: Face Detected")
            else:
                self.name_label.config(text="Detected: None")
                self.status.config(text="Status: Waiting...")

            # ===== SHOW FRAME =====
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)


# ===== RUN =====
root = tk.Tk()
app = App(root)
root.mainloop()