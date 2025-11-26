import asyncio
from os import wait
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
        self.buf: List[NDArray[np.float64]] = []


    def _process_audio(
        self,
        indata: NDArray[np.float64],
        frames: int,
        time_info,
        status
    ) -> None:
        required_samples = int(self.samplerate * self.sample_duration_s)

        if status:
            print(status)
        if indata.ndim > 1:
            audio_1d = indata[:, 0]  # Take first channel
        else:
            audio_1d = indata

        audio_1d = audio_1d.flatten()

        self.buf.append(audio_1d)
        num_samples = sum(chunk.shape[0] for chunk in self.buf)

        if num_samples >= required_samples:
            data = np.concatenate(self.buf, axis=0)
            data = data[:required_samples]

            df = eaf.extract_features_raw(data)

            if len(df) > 0:
                self.process(df)
            else:
                print("couldn't make a prediction")

            self.buf = []


    def run(self) -> None:
        stream = sd.InputStream(
            callback=self._process_audio,
            blocksize=self.blocksize,
            channels=self.channels,
            samplerate=self.samplerate
        )

        stream.start()

