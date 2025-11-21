import sys
import sounddevice as sd
import soundfile as sf
from joblib import load
from pathlib import Path

# add src as module for importing
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "data" / "audio" / "live"
sys.path.append(str(PROJECT_ROOT / "src"))
from speech_recognition.utils import extract_audio_features as eaf


def record(duration=1.0):
    sr = eaf.SAMPLE_RATE
    out_file = DEMO_DIR / "demo.wav"
    recording = sd.rec(int(sr * duration), samplerate=sr, channels=1)
    sd.wait()                          # wait for recording to finish
    sf.write(out_file, recording, sr)  # overwritten each loop
    return out_file


def main():
    pipeline = load(PROJECT_ROOT / "notebooks" / "speech_rec_model.pkl")
    index_2_speaker = {
        0: "Drake",
        1: "Melissa",
        2: "Lisa",
        3: "Dan",
        4: "David"
    }

    while True:
        # record 1 sec of audio
        wav_file = record(2.0)
        df = eaf.extract_features(wav_file, "unknown")

        # if empty -> no audio was recorded -> continue
        if df.empty:
            print("No Speaker Detected")
        else:
            # if audio recorded -> add feature names, drop target, and then use model to predict speaker
            df.columns = eaf.get_feature_names()
            xs = df.drop(columns=['speaker'])
            print(df.to_string())

            probs = pipeline.predict_proba(xs)
            avg = probs.mean(axis=0)
            best_idx = avg.argmax()
            predicted = index_2_speaker[pipeline.classes_[best_idx]]
            certainty = avg[best_idx]
            print(f"Speaker: {predicted} | Certainty: {certainty:.2f}")
            break

if __name__ == '__main__':
    main()
