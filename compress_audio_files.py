import os
import subprocess


def compress_audio():
    input_dir = "./audio"
    output_dir = "./audio_16k"

    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)

        for file in files:
            if file.endswith(".wav"):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_subdir, file)

                command = [
                    "ffmpeg",
                    "-i",
                    input_file,
                    "-ar",
                    "16000",
                    output_file,
                    "-y"
                ]
                subprocess.run(command)

    print("Conversion completed!")


if __name__ == "__main__":
    compress_audio()
