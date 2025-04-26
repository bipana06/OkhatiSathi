import random

# Path to your CSV file
csv_path = "metadata.csv"  # Replace with your actual CSV path

# Output files
train_file = "train_list.txt"
val_file = "val_list.txt"

# Load and shuffle the data
with open(csv_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    random.shuffle(lines)

# Split 95% train, 5% validation
split_idx = int(0.9 * len(lines))
train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

# Save to files
with open(train_file, 'w', encoding='utf-8') as f:
    f.writelines(train_lines)

with open(val_file, 'w', encoding='utf-8') as f:
    f.writelines(val_lines)

print(f"Done! Saved {len(train_lines)} training samples to {train_file}")
print(f"Done! Saved {len(val_lines)} validation samples to {val_file}")
