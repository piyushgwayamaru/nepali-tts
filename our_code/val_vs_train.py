import pandas as pd
import matplotlib.pyplot as plt

# Load training and validation loss CSVs
train_df = pd.read_csv(r"E:\newtacotron\tacotron\our_code\step_loss.csv")
val_df = pd.read_csv(r'E:\newtacotron\tacotron\our_code\smoothed_val_loss_rounded.csv')

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(train_df["step_name"], train_df["loss"], label="Training Loss", color='blue')
plt.plot(val_df["step_name"], val_df["smoothed_val_loss"], label="Validation Loss (Smoothed)", color='red')
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as PNG
plt.savefig(r"E:\newtacotron\tacotron\our_code\loss_curve.png", dpi=300)

# Show the plot
plt.show()
