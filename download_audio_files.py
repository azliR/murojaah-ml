from __future__ import annotations
import csv

import requests
import os
from tqdm import tqdm
from urllib.parse import urlparse


def prepare_recording_directory(directory_path: str, surah_num: int, ayah_num: int) -> str:
    recording_directory_path = os.path.join(directory_path, 'surah_{}'.format(surah_num), 'ayah_{}'.format(ayah_num))
    os.makedirs(recording_directory_path, exist_ok=True)
    return recording_directory_path


def read_csv(csv_path: str) -> list[list[str]]:
    with open(csv_path, 'r') as file:
        content = file.read()
    csv_data = csv.reader(content.splitlines(), delimiter=',')
    entries = list(csv_data)
    header_row = entries[0]
    dataset = entries[1:]
    labeled_entries = [
        entry for entry in dataset if entry[9] == 'True' and entry[10] == 'True'
    ]
    print('Total data: {}'.format(len(dataset)))
    print('Total labeled data: {}'.format(len(labeled_entries)))
    return labeled_entries


def download_recording_from_url(url_str: str, download_folder: str, use_cache: bool = True) -> str | None:
    parsed_url = urlparse(url_str)
    wav_filename = os.path.basename(parsed_url.path)

    download_file_path = os.path.join(download_folder, wav_filename)

    if not os.path.exists(download_file_path) or not use_cache:
        r = requests.get(url_str, allow_redirects=True)
        if r.status_code == 200:
            open(download_file_path, 'wb').write(r.content)
        else:
            print('Failed to download {} with status {}'.format(wav_filename, r.status_code))
            return None
    else:
        print('File {} already exists'.format(wav_filename))

    return download_file_path


def download_all_recordings(labeled_entries: list[list[str]]) -> None:
    succeed_download = 0
    for entry in tqdm(labeled_entries, desc='Audio Files'):
        url = entry[2]
        print(url)
        downloaded_recording_dir = prepare_recording_directory('audio', int(entry[0]), int(entry[1]))
        result = download_recording_from_url(url, downloaded_recording_dir, use_cache=False)
        if result is not None:
            succeed_download = succeed_download + 1
    print('Succeed download: {}'.format(succeed_download))


if __name__ == "__main__":
    labeled_entries = read_csv('./data/murojaah-ml_v1.0_labeled.csv')
    download_all_recordings(labeled_entries)
