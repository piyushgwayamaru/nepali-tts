import matplotlib.pyplot as plt
import csv
import numpy as np

# Input file prath
input_file = r"E:\newtacotron\tacotron\our_code\validation_loss.csv"  # Replace with the path to your CSV file
output_plot = r"E:\newtacotron\tacotron\our_code\validation_loss.png"

# Lists to store steps and losses
steps = []
losses = []

# Read the CSV file
try:
    with open(input_file, "r") as file:
        reader = csv.reader(file)
        # Skip the header
        next(reader)
        # Extract step and loss values
        for row in reader:
            if len(row) >= 2:  # Ensure the row has at least two columns
                try:
                    steps.append(int(row[0]))  # Convert step to integer
                    losses.append(float(row[1]))  # Convert loss to float
                except ValueError:
                    print(f"Warning: Skipping invalid row: {row}")
except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
    exit(1)
except Exception as e:
    print(f"Error reading the file: {e}")
    exit(1)

# Check if data was extracted
if not steps or not losses:
    print("Error: No valid data found in the file.")
    exit(1)

# Determine plotting style based on the number of steps
num_steps = len(steps)
use_markers = num_steps < 1000  # Use markers for small datasets (<1000 steps)

# Create the step vs. loss plot
plt.figure(figsize=(10, 6))  # Compact figure size
if use_markers:
    # For small datasets, plot with markers and line
    plt.plot(steps, losses, marker='o', linestyle='-', color='b', markersize=4, linewidth=1)
else:
    # For large datasets, plot line only to avoid clutter
    plt.plot(steps, losses, linestyle='-', color='b', linewidth=1)

# Customize the plot
plt.title('Step vs. Loss Curve', fontsize=14)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Dynamically adjust x-axis ticks for readability
if num_steps > 10:
    # Use fewer ticks for large datasets
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10, integer=True))
else:
    # Use all steps for small datasets
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Adjust y-axis to focus on loss range
plt.margins(y=0.1)  # Add 10% margin around the loss values
plt.tight_layout()

# Save the plot to a file
try:
    plt.savefig(output_plot, dpi=300)  # Higher DPI for clarity
    print(f"Plot saved as {output_plot}")
except Exception as e:
    print(f"Error saving the plot: {e}")

# Display the plot
plt.show()