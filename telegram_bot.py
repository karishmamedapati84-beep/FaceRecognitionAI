import requests
from config import TELEGRAM_TOKEN, CHAT_ID


def send_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text}

    res = requests.post(url, data=data)
    print("Message response:", res.text)


def send_video(video_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo"

    try:
        with open(video_path, "rb") as video:
            files = {"video": video}
            data = {"chat_id": CHAT_ID}

            res = requests.post(url, files=files, data=data)

            print("VIDEO RESPONSE:", res.text)   # 🔥 VERY IMPORTANT

    except Exception as e:
        print("VIDEO ERROR:", e)