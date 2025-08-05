import csv

# Input CSV and output TSV paths
input_csv = r"E:\newtacotron\tacotron\nepali\manifest.csv"
output_tsv = r"E:\newtacotron\tacotron\nepali\line_index.tsv"

def csv_to_tsv_replace_commas(input_path, output_path):
    rows = []

    # Read the CSV file and process rows
    with open(input_path, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        for row in reader:
            # Replace commas in each field with space
            new_row = [field.replace(",", " ") for field in row]
            rows.append(new_row)

    # Write the processed rows to a TSV file
    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        writer.writerows(rows)

    print(f"✅ Converted CSV to TSV: {output_path}")

# Run the function
csv_to_tsv_replace_commas(input_csv, output_tsv)
