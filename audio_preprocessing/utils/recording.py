import os
import numpy as np
from typing import Optional

ALL_SURAHS = None
NUM_SURAHS = 114


def get_paths_to_surah_recordings(local_download_dir: str, surahs: Optional[int] = ALL_SURAHS):
    paths_to_audio = []
    if not os.path.isdir(local_download_dir):
        raise OSError('Local download directory {} not found'.format(local_download_dir))
    if surahs is None:
        surahs = 1 + np.arange(NUM_SURAHS)
    for surah_num in surahs:
        local_surah_dir = os.path.join(local_download_dir, "surah_" + str(surah_num))
        for _, ayah_directories, _ in os.walk(local_surah_dir):
            for ayah_directory in ayah_directories:
                local_ayah_dir = os.path.join(local_surah_dir, ayah_directory)

                for _, _, recording_filenames in os.walk(local_ayah_dir):
                    for recording_filename in recording_filenames:
                        local_audio_path = os.path.join(local_ayah_dir, recording_filename)

                        paths_to_audio.append(local_audio_path)

    return paths_to_audio
