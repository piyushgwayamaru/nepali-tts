import pandas as pd

# Load the CSV file
df = pd.read_csv(r'E:\newtacotron\tacotron\our_code\step_loss(16k).csv')  # Replace with your actual file path

# Select the 'loss' column
loss_values = df['loss']

# Calculate the average loss
average_loss = loss_values.mean()

print(f"Average loss: {average_loss:.5f}")
