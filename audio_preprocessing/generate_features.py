import functools
import os

import librosa
import numpy as np
import tensorflow as tf
from tensorflow import signal as tf_signal

import utils.recording as recording_utils

ALL_SURAHS = 0
NUM_SURAHS = 114

FRAME_SIZE_S = 0.025
FRAME_STRIDE_S = 0.01

pre_emphasis_factor_1 = 0.95
pre_emphasis_factor_2 = 0.97
PRE_EMPHASIS_FACTOR = pre_emphasis_factor_1

STFT_NUM_POINTS_1 = 256
STFT_NUM_POINTS_2 = 512
STFT_NUM_POINTS = STFT_NUM_POINTS_2

NUM_TRIANGULAR_FILTERS = 40
NUM_MFCCS = 13

SUPPORTED_FREQUENCIES = [8000, 16000, 32000, 44100, 48000]


def generate_mel_filter_banks(signal, sample_rate_hz, frame_size_s=FRAME_SIZE_S, frame_stride_s=FRAME_STRIDE_S,
                              window_fn=functools.partial(tf_signal.hamming_window, periodic=True),
                              fft_num_points=STFT_NUM_POINTS, lower_freq_hz=0.0, num_mel_bins=NUM_TRIANGULAR_FILTERS,
                              log_offset=1e-6, should_log_weight=False):
    signal = tf.convert_to_tensor(signal, dtype=tf.float32)

    frame_length = int(sample_rate_hz * frame_size_s)
    frame_step = int(sample_rate_hz * frame_stride_s)

    upper_freq_hz = sample_rate_hz / 2.0

    frames = tf_signal.frame(signal, frame_length=frame_length, frame_step=frame_step, pad_end=True, pad_value=0)

    stfts = tf_signal.stft(frames,
                           frame_length=frame_length,
                           frame_step=frame_step,
                           fft_length=fft_num_points,
                           window_fn=window_fn)

    magnitude_spectrograms = tf.abs(stfts)
    power_spectograms = tf.math.real(stfts * tf.math.conj(stfts))

    num_spectrogram_bins = 1 + int(fft_num_points / 2)

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=num_mel_bins,
                                                                        num_spectrogram_bins=num_spectrogram_bins,
                                                                        sample_rate=sample_rate_hz,
                                                                        lower_edge_hertz=lower_freq_hz,
                                                                        upper_edge_hertz=upper_freq_hz,
                                                                        dtype=tf.float32)

    mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    if should_log_weight:
        return tf.math.log(mel_spectrograms + log_offset)
    else:
        return mel_spectrograms


def generate_log_mel_filter_banks(signal, sample_rate_hz, frame_size_s=FRAME_SIZE_S, frame_stride_s=FRAME_STRIDE_S,
                                  window_fn=functools.partial(tf_signal.hamming_window, periodic=True),
                                  fft_num_points=STFT_NUM_POINTS, lower_freq_hz=0.0,
                                  num_mel_bins=NUM_TRIANGULAR_FILTERS, log_offset=1e-6):
    return generate_mel_filter_banks(signal, sample_rate_hz, frame_size_s=frame_size_s, frame_stride_s=frame_stride_s,
                                     window_fn=window_fn, fft_num_points=fft_num_points,
                                     lower_freq_hz=lower_freq_hz, num_mel_bins=num_mel_bins, log_offset=log_offset,
                                     should_log_weight=True)


def generate_mfcc(signal, sample_rate_hz, num_mfccs=NUM_MFCCS, frame_size_s=FRAME_SIZE_S, frame_stride_s=FRAME_STRIDE_S,
                  window_fn=functools.partial(tf_signal.hamming_window, periodic=True), fft_num_points=STFT_NUM_POINTS,
                  lower_freq_hz=0.0, num_mel_bins=NUM_TRIANGULAR_FILTERS, log_offset=1e-6, should_log_weight=True):
    log_mel_filter_banks = \
        generate_log_mel_filter_banks(signal, sample_rate_hz, frame_size_s=frame_size_s, frame_stride_s=frame_stride_s,
                                      window_fn=window_fn, fft_num_points=fft_num_points, lower_freq_hz=lower_freq_hz,
                                      num_mel_bins=num_mel_bins, log_offset=log_offset)

    return tf_signal.mfccs_from_log_mel_spectrograms(log_mel_filter_banks)[..., :num_mfccs]


def generate_features():
    surahs_to_tensorize = np.arange(NUM_SURAHS) + 1

    print(surahs_to_tensorize)
    paths_to_tensorize = recording_utils.get_paths_to_surah_recordings("../audio", surahs_to_tensorize)

    print(paths_to_tensorize)

    for path in paths_to_tensorize:
        audio_data, sample_rate_hz = librosa.load(path)

        if audio_data.shape[0] < int(FRAME_SIZE_S * sample_rate_hz):
            print('Recording at path {} is not long enough.'.format(path))
            continue

        audio_data = audio_data.transpose()
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate_hz, target_sr=16000)
        sample_rate_hz = 16000

        if sample_rate_hz in SUPPORTED_FREQUENCIES:
            output = generate_mfcc(audio_data, sample_rate_hz)
        else:
            print('Unsupported sampling frequency for recording at path %s: %d.' % (
                path, sample_rate_hz))
            continue

        path1, base_filename = os.path.split(path)
        filename, _ = os.path.splitext(base_filename)

        filename += "_" + str(sample_rate_hz)

        path2, ayah_folder = os.path.split(path1)
        _, surah_folder = os.path.split(path2)

        save_dir = os.path.join("../data", "mfcc", surah_folder, ayah_folder)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, filename)
        output_np_array = output.numpy()
        np.save(save_path, output_np_array)


if __name__ == "__main__":
    generate_features()
