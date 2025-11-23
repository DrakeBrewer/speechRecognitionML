import sounddevice as sd
import soundfile as sf
from pathlib import Path
import argparse


PROJECT_ROOT_DIR = Path(__file__).resolve().parents[3]
AUDIO_DIR = PROJECT_ROOT_DIR / "data" / "audio" / "raw"
SAMPLE_RATE = 22050


def record(duration=120.0, file_name="unknown_01"):
    out_file = AUDIO_DIR / f"{file_name}"

    # Warm-up recording
    sd.rec(int(0.05 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()

    # Real recording
    print(f"Recording {file_name} for {duration} seconds ...")
    recording = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()

    sf.write(out_file, recording, SAMPLE_RATE, subtype='PCM_16')
    print(f"Saved recording to {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Record audio")
    parser.add_argument("--speaker", type=str, help="Speaker name")
    parser.add_argument("--session", type=str, help="Session ID, ex: 01, 02")
    parser.add_argument("--duration", type=float, default=120.0, help="Recording duration in seconds")

    arguments = parser.parse_args()
    filename = f"{arguments.speaker}_{arguments.session}.wav"
    record(arguments.duration, filename)


if __name__ == '__main__':
    main()