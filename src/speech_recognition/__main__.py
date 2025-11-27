from typing import Tuple
import pandas as pd
from joblib import load
from pathlib import Path
from speech_recognition.utils.audio_stream import Audio_Processor
from speech_recognition.utils.window import Window

PROJECT_ROOT = Path(__file__).resolve().parents[2]

last_prediction: Tuple[str, float] = ("Unrecognized", 0.0)

def main():
    global last_prediction
    model = load(PROJECT_ROOT / "notebooks" / "speech_rec_model.pkl")
    index_2_speaker = {
        0: "Drake",
        1: "Melissa",
        2: "Lisa",
        3: "Dan",
        4: "David"
    }

    def process_features(df: pd.DataFrame) -> None:
        global last_prediction
        probs = model.predict_proba(df)
        avg = probs.mean(axis=0)
        best_idx = avg.argmax()
        predicted = index_2_speaker[model.classes_[best_idx]]
        certainty = avg[best_idx]

        if certainty < 0.65:
            predicted = "Unrecognized"

        last_prediction = (predicted, certainty)

    def get_last_prediction():
        return last_prediction

    stream = Audio_Processor(
        blocksize=1024,
        channels=1,
        samplerate=22050,
        sample_duration_s=1,
        process=process_features,
    )
    stream.run()

    window = Window(
        title="speech recognizer",
        dimensions="300x200",
        bg_color="#424242",
        update_func=get_last_prediction
    )
    window.run()

if __name__ == "__main__":
    main()
