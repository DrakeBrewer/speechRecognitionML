import asyncio
from pathlib import Path
import sys
from typing import Any, AsyncGenerator, List, Tuple
from joblib import load
import sounddevice as sd
import numpy as np
from numpy.typing import NDArray
from speech_recognition.utils import extract_audio_features as eaf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
model = load(PROJECT_ROOT / "notebooks" / "speech_rec_model.pkl")
index_2_speaker = {
    0: "Drake",
    1: "Melissa",
    2: "Lisa",
    3: "Dan",
    4: "David"
}

async def inputstream_generator(
            blocksize=1024,
            channels=1,
            samplerate=22050
    ) -> AsyncGenerator[Tuple[NDArray[np.float64], Any]]:

    q_in: asyncio.Queue[Tuple[NDArray[np.float64], Any]] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(
            indata: NDArray[np.float64],
            frame_count: int,
            time_info: Any,
            status: int
    ):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(callback=callback, blocksize=blocksize, channels=channels, samplerate=samplerate)
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


async def process_audio(blocksize: int, channels: int, samplerate: int) -> None:
    buf: List[NDArray[np.float64]] = []
    duration = 1.0
    required_samples = int(samplerate * duration)

    async for indata, status in inputstream_generator(blocksize, channels, samplerate):
        if status:
            print(status)
        if indata.ndim > 1:
            audio_1d = indata[:, 0]  # Take first channel
        else:
            audio_1d = indata

        audio_1d = audio_1d.flatten()

        buf.append(audio_1d)
        num_samples = sum(chunk.shape[0] for chunk in buf)

        if num_samples >= required_samples:
            data = np.concatenate(buf, axis=0)
            data = data[:required_samples]

            df = eaf.extract_features_raw(data)

            if len(df) > 0:
                probs = model.predict_proba(df)
                avg = probs.mean(axis=0)
                best_idx = avg.argmax()
                predicted = index_2_speaker[model.classes_[best_idx]]
                certainty = avg[best_idx]
                print(f"Speaker: {predicted} | Certainty: {certainty:.2f}")
            else:
                print("couldn't make a prediction")

            buf = []


async def main():
    audio_task = asyncio.create_task(process_audio(blocksize=22050, channels=1, samplerate=22050))
    try:
        await audio_task
    except asyncio.CancelledError:
        print('\nwire was cancelled')

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
