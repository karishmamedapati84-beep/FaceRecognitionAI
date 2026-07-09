from deepface import DeepFace
import os

MIN_ACC = 75  # balanced for real-world


def recognize_face(frame):
    try:
        result = DeepFace.find(
            img_path=frame,
            db_path="dataset/",
            enforce_detection=False,
            model_name="Facenet512"
        )

        if len(result) > 0 and len(result[0]) > 0:
            df = result[0]

            # ✅ Best match
            best_match = df.loc[df['distance'].idxmin()]

            identity = best_match['identity']
            distance = best_match['distance']

            accuracy = int((1 - distance) * 100)

            print("DEBUG →", identity, accuracy)

            if accuracy >= MIN_ACC:
                name = os.path.basename(os.path.dirname(identity))
                return name, accuracy

    except Exception as e:
        print("Error:", e)

    return "Unknown", 0