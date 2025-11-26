import asyncio
import numpy as np
import pandas as pd
import sounddevice as sd
from typing import Callable
from numpy.typing import NDArray
from typing import Any, List, Tuple
from speech_recognition.utils import extract_audio_features as eaf


class Audio_Processor:
    def __init__(
        self,
        blocksize: int,
        channels: int,
        samplerate: int,
        sample_duration_s: int,
        process: Callable[[pd.DataFrame], None]
    ):
        self.blocksize = blocksize
        self.channels = channels
        self.samplerate = samplerate
        self.sample_duration_s = sample_duration_s

        self.process = process

    async def _inputstream_generator(self):
        q_in: asyncio.Queue[Tuple[NDArray[np.float64], Any]] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def callback(
                indata: NDArray[np.float64],
                frame_count: int,
                time_info: Any,
                status: int
        ):
            loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

        stream = sd.InputStream(
            callback=callback,
            blocksize=self.blocksize,
            channels=self.channels,
            samplerate=self.samplerate
        )
        with stream:
            while True:
                indata, status = await q_in.get()
                yield indata, status

    async def _process_audio(self) -> None:
        buf: List[NDArray[np.float64]] = []
        duration = 1.0
        required_samples = int(self.samplerate * duration)

        async for indata, status in self._inputstream_generator():
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
                    self.process(df)
                else:
                    print("couldn't make a prediction")

                buf = []

    async def run(self):
        await asyncio.create_task(self._process_audio())

