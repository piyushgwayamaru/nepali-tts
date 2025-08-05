import os
import shutil

def separate_spectrograms(source_dir, mel_dir, spec_dir):
    # Create target directories if they don't exist
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.endswith(".npy"):
            if filename.startswith("nepali-mel-"):
                shutil.copy2(os.path.join(source_dir, filename), os.path.join(mel_dir, filename))
            elif filename.startswith("nepali-spec-"):
                shutil.copy2(os.path.join(source_dir, filename), os.path.join(spec_dir, filename))

    print("✅ Separation complete!")

# === USAGE ===
source_folder = r"E:\newtacotron\tacotron\16k"        # Where all .npy files are currently
mel_output_folder = r"E:\newtacotron\tacotron\our_code\mel_output"   # Target folder for mel files
spec_output_folder = r"E:\newtacotron\tacotron\our_code\spec_output" # Target folder for spec files

separate_spectrograms(source_folder, mel_output_folder, spec_output_folder)
