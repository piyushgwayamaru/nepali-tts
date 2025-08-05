import re

# Input and output file paths
input_file = r"E:\newtacotron\tacotron\logs-tacotron\train.log"  # Replace with the path to your train.log file
output_file = r"E:\newtacotron\tacotron\our_code\step_loss.csv"

# Regular expression to match lines with step and loss (e.g., "Step 501 ... loss=0.20500, ...")
pattern = r"Step (\d+)\s+.*loss=([\d.]+)"

# Lists to store extracted data
steps = []
losses = []

# Read the input log file
try:
    with open(input_file, "r") as file:
        for line in file:
            # Search for the pattern in each line
            match = re.search(pattern, line)
            if match:
                # Extract step number and loss value
                step = match.group(1)
                loss = match.group(2)
                steps.append(step)
                losses.append(loss)
except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
    exit(1)
except Exception as e:
    print(f"Error reading the file: {e}")
    exit(1)

# Write the extracted data to a CSV file
try:
    with open(output_file, "w") as file:
        # Write the header
        file.write("step_name, loss\n")
        # Write each step and loss pair
        for step, loss in zip(steps, losses):
            file.write(f"{step}, {loss}\n")
    print(f"Successfully wrote data to {output_file}")
except Exception as e:
    print(f"Error writing to {output_file}: {e}")