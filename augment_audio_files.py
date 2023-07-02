import os
import soundfile as sf
from tqdm import tqdm
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, TimeMask, TanhDistortion


def augment_audio(input_file, output_file):
    samples, sample_rate = sf.read(input_file)

    transforms = [
        AddGaussianNoise(p=1),
        TimeStretch(p=1),
        PitchShift(p=1),
        TanhDistortion(p=1),
    ]
    transforms_names = [
        "noised",
        "time_stretched",
        "pitch_shifted",
        "tanh_distorted",
    ]

    sf.write(output_file, samples, sample_rate)
    for i, transform in enumerate(transforms):
        augment = Compose([transform])
        new_output_file = output_file[:-4] + "_{}.wav".format(transforms_names[i])
        augmented_samples = augment(samples=samples, sample_rate=16000)
        sf.write(new_output_file, augmented_samples, sample_rate)


def main():
    input_dir = "./audio_16k"
    output_dir = "./audio_augmented"

    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)

        for file in tqdm(files):
            if file.endswith(".wav"):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_subdir, file)
                augment_audio(input_file, output_file)

    print("Conversion completed!")


if __name__ == "__main__":
    main()
