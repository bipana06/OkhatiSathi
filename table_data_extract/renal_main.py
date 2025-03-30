import pdfplumber
import pandas as pd

pdf_path = "renal.pdf"
output_csv = "renal_output.csv"

all_tables = []

header = ["Drug", "Grade", "Comment"]
header_seen = False  # Track if the header has been added
previous_first = ""  # Store the first element of the previous row

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            clean_table = []
            for row in table:
                row = [col.strip().replace("\n", " ") if col else "" for col in row]  # Remove newlines, strip spaces

                # Skip header repetition
                if row == header:
                    continue
                
                # If the first column is empty, use the previous row’s first column
                if len(row) > 0 and row[0] == "":
                    row[0] = previous_first
                
                clean_table.append(row)  # Keep the row as is

                # Store first element for the next iteration
                if len(row) > 0:
                    previous_first = row[0]

            if clean_table:
                df = pd.DataFrame(clean_table, columns=header[:len(clean_table[0])])  # Keep only necessary column names
                
                if not header_seen:
                    all_tables.append(df)  # Add first table with headers
                    header_seen = True
                else:
                    all_tables.append(df.iloc[1:])  # Skip duplicate headers in subsequent tables

# Combine all extracted tables and save to CSV
if all_tables:
    final_df = pd.concat(all_tables, ignore_index=True)
    final_df.to_csv(output_csv, index=False)

print(f"Cleaned table saved to {output_csv}")
