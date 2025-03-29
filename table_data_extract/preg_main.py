import pdfplumber
import pandas as pd

pdf_path = "preg.pdf"
output_csv = "preg_output.csv"

all_tables = []
header = ["Medicine", "Comment", "Cat."]
header_seen = False  # Track if the header has already been added
previous_row = None  # Store the previous row's data to handle missing values

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            clean_table = []
            for row in table:
                row = [col.strip().replace("\n", " ") if col else "" for col in row]  # Remove newlines, strip spaces

                # Handle case where a row has exactly 2 elements and the first element is empty
                if len(row) == 2:
                    if row[0] == "" and previous_row:
                        # Use the first element from the previous row
                        row[0] = previous_row[0]
                
                # Skip rows that don't have enough actual values
                if len(row) < 3 or sum(1 for col in row if col.strip()) < 3:
                    continue  

                if row == header:  
                    # Skip redundant headers
                    continue
                
                clean_table.append(row[:3])  # Ensure exactly 3 elements per row

                # Store the current row to use in the next iteration
                previous_row = row

            if clean_table:
                df = pd.DataFrame(clean_table, columns=header)
                
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

