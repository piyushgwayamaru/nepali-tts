def extract_transcripts_formatted(input_file, output_file, num_lines=None):
    """
    Extract transcripts and save as quoted, comma-separated lines:
    'text1',
    'text2',
    ...
    """
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for i, line in enumerate(fin):
            if num_lines is not None and i >= num_lines:
                break

            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                transcript = parts[1].replace("'", "\\'")  # escape single quotes in text if any
                fout.write(f"'{transcript}',\n")
            else:
                fout.write("'',\n")  # empty string if format unexpected

    print(f"Processed {i+1 if num_lines is None else min(i+1, num_lines)} lines.")
    print(f"Formatted transcripts saved to '{output_file}'")

# === USAGE ===
input_path = r"E:\newtacotron\tacotron\nepali\line_index.tsv"
output_path =r"E:\newtacotron\tacotron\our_code\transcripts.tsv"
lines_to_process = 1000  # or an integer number of lines to process

extract_transcripts_formatted(input_path, output_path, num_lines=lines_to_process)
