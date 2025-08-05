import pandas as pd

# Load the first CSV
df = pd.read_csv(r"E:\newtacotron\tacotron\our_code\validation_loss.csv")  # Replace with your actual path

# Round to 5 decimal places
df["smoothed_val_loss"] = df["smoothed_val_loss"].round(5)

# Save the result to a new file (or overwrite the same one)
df.to_csv("smoothed_val_loss_rounded.csv", index=False)
