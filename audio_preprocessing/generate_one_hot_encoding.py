import json

import dill as pickle
import numpy as np

QURAN_KEY = "quran"
SURAHS_KEY = "surahs"
AYAHS_KEY = "ayahs"
TEXT_KEY = "text"

NUM_KEY = "num"
NAME_KEY = "name"
BISMILLAH_KEY = "bismillah"

ENCODING_MAP_KEY = "encoding_map"
DECODING_MAP_KEY = "decoding_map"
CHAR_TO_INT_MAP_KEY = "char_to_int"
INT_TO_CHAR_MAP_KEY = "int_to_char"


def create_list_of_quranic_chars(quran_obj, surahs_key=SURAHS_KEY, ayahs_key=AYAHS_KEY, text_key=TEXT_KEY):
    quranic_char_set = set()

    for surah_obj in quran_obj[surahs_key]:
        for ayah_obj in surah_obj[ayahs_key]:
            ayah_text = ayah_obj[text_key]

            for char in ayah_text:
                quranic_char_set.add(char)

    return sorted(list(quranic_char_set))


def create_one_hot_encoding(quranic_char_list):
    char_to_int = dict((c, i) for i, c in enumerate(quranic_char_list))
    int_to_char = dict((i, c) for i, c in enumerate(quranic_char_list))

    def encode_char_as_one_hot(string, char_to_int):
        str_len = len(string)
        int_list = np.array([char_to_int[char] for char in string])

        one_hot_string = np.zeros((str_len, len(char_to_int)))
        one_hot_string[np.arange(str_len), int_list] = 1

        return one_hot_string

    def decode_one_hot_as_string(one_hot_string, int_to_char):
        int_list = list(np.argmax(one_hot_string, axis=1))
        char_list = [int_to_char[integer] for integer in int_list]

        return str(char_list)

    return char_to_int, int_to_char, encode_char_as_one_hot, decode_one_hot_as_string


def generate_a_one_hot_encoded_script(quran_obj,
                                      encoding_fn,
                                      surahs_key=SURAHS_KEY,
                                      ayahs_key=AYAHS_KEY,
                                      text_key=TEXT_KEY,
                                      num_key=NUM_KEY,
                                      name_key=NAME_KEY,
                                      bismillah_key=BISMILLAH_KEY):
    one_hot_quran_encoding = {}
    one_hot_quran_encoding[SURAHS_KEY] = []

    for surah_obj in quran_obj[surahs_key]:
        # Copy new surah object for one-hot Json container.
        one_hot_surah_obj = {}
        one_hot_surah_obj[num_key] = surah_obj[num_key]
        one_hot_surah_obj[name_key] = surah_obj[name_key]
        one_hot_surah_obj[ayahs_key] = []

        for ayah_obj in surah_obj[ayahs_key]:
            ayah_text = ayah_obj[text_key]

            # Make new ayah object for one-hot Json container.
            one_hot_ayah_obj = {}
            one_hot_ayah_obj[num_key] = ayah_obj[num_key]
            one_hot_ayah_obj[text_key] = encoding_fn(ayah_text)

            if bismillah_key in ayah_obj:
                one_hot_ayah_obj[bismillah_key] = encoding_fn(ayah_obj[bismillah_key])

            one_hot_surah_obj[ayahs_key].append(one_hot_ayah_obj)
        one_hot_quran_encoding[surahs_key].append(one_hot_surah_obj)

    return one_hot_quran_encoding


if __name__ == "__main__":
    with open('../data/data-uthmani.json', 'rb') as quran_json_file:
        quran_obj = json.load(quran_json_file)[QURAN_KEY]

    quranic_char_list = create_list_of_quranic_chars(quran_obj)
    print(quranic_char_list, ' has ', len(quranic_char_list), ' characters.')

    char_to_int_map, int_to_char_map, \
        encode_char_as_one_hot, \
        decode_one_hot_as_string = create_one_hot_encoding(quranic_char_list)

    print("Successfully encoded!")
    print(char_to_int_map)

    one_hot_quran_encoding = generate_a_one_hot_encoded_script(
        quran_obj,
        lambda string: encode_char_as_one_hot(string, char_to_int_map))

    full_object = {
        QURAN_KEY: one_hot_quran_encoding,
        ENCODING_MAP_KEY: encode_char_as_one_hot,
        DECODING_MAP_KEY: decode_one_hot_as_string,
        CHAR_TO_INT_MAP_KEY: char_to_int_map,
        INT_TO_CHAR_MAP_KEY: int_to_char_map
    }

    with open("../data/one-hot.pkl", 'wb') as one_hot_quran_pickle_file:
        pickle.dump(full_object, one_hot_quran_pickle_file)
