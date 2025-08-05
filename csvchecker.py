import csv

def remove_duplicate_voice_ids(input_csv_path, output_csv_path):
    seen = set()
    cleaned_rows = []

    # Read and filter
    with open(input_csv_path, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if not row or len(row) < 2:
                continue
            voice_id = row[0].strip()
            if voice_id not in seen:
                seen.add(voice_id)
                cleaned_rows.append([voice_id, ','.join(row[1:]).strip()])

    # Write cleaned CSV
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)

    print(f"✅ Cleaned CSV written to: {output_csv_path}")
    print(f"🧹 Removed {len(seen) - len(cleaned_rows)} duplicate rows." if len(seen) != len(cleaned_rows) else "✅ No duplicates found.")

# Example usage
if __name__ == "__main__":
    input_csv = r"E:\newtacotron\tacotron\nepali\eval.csv"
    output_csv = r"E:\newtacotron\tacotron\nepali\cleaned.csv"

    remove_duplicate_voice_ids(input_csv, output_csv)
