import pdfplumber
import pandas as pd

pdf_path = "breast.pdf"
output_csv = "breast_output.csv"

all_tables = []
header = ["Drug", "Comments"]  # Only 2 columns needed
header_seen = False  # Track if the header has already been added

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            clean_table = []
            for row in table:
                row = [col.strip().replace("\n", " ") if col else "" for col in row]  # Remove newlines, strip spaces
                
                # Ensure exactly 2 elements per row
                row = row[:2]  

                # Skip rows with less than 2 actual values
                if sum(1 for col in row if col.strip()) < 2:
                    continue  

                # Skip redundant headers appearing later in the table
                if row == header:  
                    continue

                clean_table.append(row)

            if clean_table:
                df = pd.DataFrame(clean_table, columns=header)
                
                if not header_seen:
                    all_tables.append(df)  # Add first table with headers
                    header_seen = True
                else:
                    all_tables.append(df)  # Append without removing any rows

# Combine all extracted tables and save to CSV
if all_tables:
    final_df = pd.concat(all_tables, ignore_index=True)
    final_df.to_csv(output_csv, index=False)

print(f"Cleaned table saved to {output_csv}")