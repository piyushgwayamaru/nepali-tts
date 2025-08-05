import os
import shutil

def group_files(input_folder, png_output_folder, wav_output_folder):
    # Ensure output directories exist
    os.makedirs(png_output_folder, exist_ok=True)
    os.makedirs(wav_output_folder, exist_ok=True)

    # List all files in the input directory
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            if filename.lower().endswith('.png'):
                shutil.move(file_path, os.path.join(png_output_folder, filename))
                print(f"Moved PNG: {filename}")
            elif filename.lower().endswith('.wav'):
                shutil.move(file_path, os.path.join(wav_output_folder, filename))
                print(f"Moved WAV: {filename}")

if __name__ == "__main__":
    input_folder = r"E:\newtacotron\tacotron\logs-tacotron"
    png_output_folder = r"E:\newtacotron\tacotron\logs-tacotron\plot"
    wav_output_folder = r"E:\newtacotron\tacotron\logs-tacotron\audio"

    group_files(input_folder, png_output_folder, wav_output_folder)
    print("File grouping completed.")

