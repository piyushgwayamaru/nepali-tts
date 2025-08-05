import os

def generate_tsv_with_custom_prefix(folder_path, output_tsv, path_prefix, max_lines=None):
    """
    Generates a TSV file listing up to max_lines .npy files in the folder,
    each line formatted as: 'path_prefix/filename',
    with a comma at the end.

    Args:
      folder_path: Folder where .npy files are located
      output_tsv: File to save output
      path_prefix: String prefix to prepend before filename in output
                   (e.g. 'E:/newtacotron/tacotron/our_code/mel_output')
      max_lines: Optional int specifying max number of lines to write. If None, write all.
    """
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])

    if max_lines is not None:
        files = files[:max_lines]

    with open(output_tsv, 'w', encoding='utf-8') as fout:
        for f in files:
            prefix = path_prefix.replace('\\', '/').rstrip('/')
            line = f"'{prefix}/{f}',\n"
            fout.write(line)

    print(f"TSV file saved to '{output_tsv}' with {len(files)} entries.")

# === USAGE ===
folder_with_mels = r"E:\newtacotron\tacotron\our_code\mel_output"
output_file = r"E:\newtacotron\tacotron\our_code\mel_output.tsv"
custom_prefix = r"E:/newtacotron/tacotron/our_code/mel_output"
num_lines_to_write = 1000  # specify how many lines you want; set to None for all

generate_tsv_with_custom_prefix(folder_with_mels, output_file, custom_prefix, max_lines=num_lines_to_write)
