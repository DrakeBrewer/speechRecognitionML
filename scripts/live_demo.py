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


def record(duration=2.0, file_name="live_demo"):
    sr = eaf.SAMPLE_RATE
    out_file = DEMO_DIR / f"{file_name}.wav"

    # Warm-up read
    sd.rec(int(0.05 * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()

    # Real recording
    print(f"Recording {file_name.name} for {duration} seconds ...")
    recording = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype="float32")
    sd.wait()

    sf.write(out_file, recording, sr, subtype='PCM_16')
    print(f"Saved recording to {out_file}")
    return out_file


def live_demo():
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
        wav_file = record(2.0, "demo")
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

def main():
    # pipeline = load(PROJECT_ROOT / "notebooks" / "speech_rec_model.pkl")
    index_2_speaker = {
        0: "Drake",
        1: "Melissa",
        2: "Lisa",
        3: "Dan",
        4: "David"
    }

    # audioFile = eaf.convert2wav(DEMO_DIR / "David_0.wav")
    # df = eaf.extract_features(PROJECT_ROOT / "data" / "audio" / "raw" / "m4a_recordings" / "Drake.wav", "Drake")
    # df.columns = eaf.get_feature_names()
    # xs = df.drop(columns=['speaker'])
    # num_preds = pipeline.predict(xs)
    # name_preds = [index_2_speaker[n] for n in num_preds]
    # df['Predicted'] = name_preds

    # correct = 0
    # for name in name_preds:
    #     if name == "David":
    #         correct += 1
    # print(df.to_string())
    # print(f"Accuracy: {correct / len(name_preds)}")



if __name__ == '__main__':
    main()
