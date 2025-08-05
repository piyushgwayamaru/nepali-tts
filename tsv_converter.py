import pandas as pd

# Load your original TSV file (with ID and text)
input_path = r'E:\newtacotron\tacotron\nepali\line_index copy.tsv'     # replace with actual path
output_path = r'E:\newtacotron\tacotron\nepali\formatted_test.tsv'  # destination TSV for evaluation

# Load with tab separator and no header
df = pd.read_csv(input_path, sep='\t', header=None, names=['utt_id', 'text'])

# Create new column 'audio_path'
df['audio_path'] = df['utt_id'].apply(lambda x: f'audio/{x}.wav')

# Select and reorder columns
df_out = df[['text', 'audio_path']]

# Save as tab-separated values
df_out.to_csv(output_path, sep='\t', index=False)
print(f'✅ Converted and saved to {output_path}')


#--> wavs folder name changed to audio
#-->