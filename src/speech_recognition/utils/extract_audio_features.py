from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import subprocess
import os
from imageio_ffmpeg import get_ffmpeg_exe



PROJECT_ROOT_DIR = Path(__file__).resolve().parents[3]
AUDIO_DIR = PROJECT_ROOT_DIR / "data" / "audio" / "raw"
OUTPUT_PATH = PROJECT_ROOT_DIR / "data"/ "audio" / "processed" / "audio_features.csv"
SAMPLE_RATE = 22050
WINDOW_LEN = 1.0
HOP_LEN = 0.5
SILENCE_THRESHOLD = 0.001


def get_feature_names():
    return (
        [f"mfcc_{i}" for i in range(1, 14)] +
        [f"chroma_{i}" for i in range(1, 13)] +
        [f"contrast_{i}" for i in range(1, 8)] +
        ["zcr", "rms"] +
        [f"mfcc_delta_{i}" for i in range(1, 14)] +
        ["speaker"]
    )


def convert2wav(in_path):
    if in_path.suffix == ".wav":                                               # already .wav
        return in_path

    out_path = in_path.with_suffix(".wav")
    ffmpeg = get_ffmpeg_exe()

    command = [
        ffmpeg, "-y",
        "-v", "quiet",
        "-i", str(in_path),
        "-ac", "1",
        "-ar", "22050",
        str(out_path)
    ]
    res = subprocess.run(command)                                              # run ffmpeg conversion util

    if (res.returncode == 0 and
            out_path.exists() and                                              # if successful -> remove old file
                out_path.stat().st_size > 0):

        print(f"{in_path.name} successfully converted to .wav")
        try:
            os.remove(in_path)
            print(f"Removed {in_path.name}")
        except OSError as err:
            print(f"Failed to remove {in_path.name}: {err}")
        return out_path                                                        # return path to wav for extraction step

    else:
        print(f"Failed to convert {in_path.name} to .wav format")
        return in_path                                                         # failed -> return original file path


def extract_features_raw(data) ->pd.DataFrame:
    sampling_rate = SAMPLE_RATE

    window_size = int(WINDOW_LEN * sampling_rate)                              # for processing small chunks at a time
    hop_size = int(HOP_LEN * sampling_rate)                                    # overlap
    feature_vectors = []                                                       # each windows (row) extracted numerical features

    frames = librosa.util.frame(data, frame_length=window_size, hop_length=hop_size)
    for i in range(frames.shape[1]):
        window = frames[:,i]

        rms = librosa.feature.rms(y=window)
        if np.mean(rms) < SILENCE_THRESHOLD:                                   # skip over silent segments
            continue

        mfcc = librosa.feature.mfcc(y=window, sr=sampling_rate, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=window, sr=sampling_rate)
        contrast = librosa.feature.spectral_contrast(y=window, sr=sampling_rate)
        zcr = librosa.feature.zero_crossing_rate(y=window)
        mfcc_delta = librosa.feature.delta(mfcc)

        feature_list = [mfcc, chroma, contrast, zcr, rms, mfcc_delta]
        features = []
        for idx, feat in enumerate(feature_list):
            if idx == 3 or idx == 4:                                           # zcr & rms are single quantifiers (ok to append)
                features.append(np.mean(feat))
            else:
                features.extend(np.mean(feat, axis=1))                         # the rest are 2D -> must extend all values

        feature_vectors.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(feature_vectors)
    feature_names = get_feature_names()
    if len(feature_names) == df.shape[1]:
        df.columns = feature_names

    return df

def extract_features(file_path, speaker: str) -> pd.DataFrame:
    print(f"Extracting features from {speaker}'s .wav file")

    mfcc_feat = librosa.feature.mfcc
    chroma_feat = librosa.feature.chroma_stft                                  # define librosa feature functions
    contrast_feat = librosa.feature.spectral_contrast
    zcr_feat = librosa.feature.zero_crossing_rate
    rms_feat = librosa.feature.rms
    mfcc_delta_feat = librosa.feature.delta

    raw_data, sampling_rate = librosa.load(file_path, sr=SAMPLE_RATE)          # raw waveform (amplitude/time)
    raw_data = raw_data / np.max(np.abs(raw_data) + 1e-6)                      # normalize

    window_size = int(WINDOW_LEN * sampling_rate)                              # for processing small chunks at a time
    hop_size = int(HOP_LEN * sampling_rate)                                    # overlap
    feature_vectors = []                                                       # each windows (row) extracted numerical features

    frames = librosa.util.frame(raw_data, frame_length=window_size, hop_length=hop_size)
    for i in range(frames.shape[1]):
        window = frames[:,i]

        rms = rms_feat(y=window)
        if np.mean(rms) < SILENCE_THRESHOLD:                                   # skip over silent segments
            continue

        mfcc = mfcc_feat(y=window, sr=sampling_rate, n_mfcc=13)                # features to be extracted from each window
        chroma = chroma_feat(y=window, sr=sampling_rate)
        contrast = contrast_feat(y=window, sr=sampling_rate)
        zcr = zcr_feat(y=window)
        mfcc_delta = mfcc_delta_feat(mfcc)

        feature_list = [mfcc, chroma, contrast, zcr, rms, mfcc_delta]
        features = []
        for idx, feat in enumerate(feature_list):
            if idx == 3 or idx == 4:                                           # zcr & rms are single quantifiers (ok to append)
                features.append(np.mean(feat))
            else:
                features.extend(np.mean(feat, axis=1))                         # the rest are 2D -> must extend all values

        features.append(speaker)                                               # add speaker name to end
        feature_vectors.append(features)

    df = pd.DataFrame(feature_vectors)
    return df


def main():
    data_frames = []
    dir_list = list(AUDIO_DIR.glob("*"))

    for file in dir_list:
        wavFile = convert2wav(file)                                            # convert each file -> .wav format
        if wavFile.suffix == ".wav":
            speaker = wavFile.stem.split("_")[0]                               # skip any files that failed to convert
            data_frames.append(extract_features(wavFile, speaker))             # extract features -> data frame

    final_df = pd.concat(data_frames, ignore_index=True)                       # combine into 1 df
    final_df.columns = get_feature_names()
    final_df.to_csv(OUTPUT_PATH, index=False)                                  # convert to csv
    print("Audio Data Successfully Processed")


if __name__ == '__main__':
    main()

