import PyPDF2
import re
import csv

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def extract_section(text, heading, next_heading):
    """Extracts multi-line content for a given heading until the next specified heading."""
    pattern = rf"{heading}:(.*?)(?=\n{next_heading}:|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

def extract_medicine_info(text):
    """Extracts medicine information and returns it as a structured list."""
    medicines = re.split(r"\n(?=[A-Z &()/-]+\n)", text)  # Split at uppercase medicine headings without numbers
    extracted_data = []

    for medicine in medicines:
        lines = medicine.strip().split("\n")
        if len(lines) < 2 or any(char.isdigit() for char in lines[0]):  # Skip page numbers
            continue

        medicine_name = lines[0].strip()

        # Extract sections using next expected heading
        dosage_form = extract_section(medicine, "Dosage form and strength", "Indication")
        indication = extract_section(medicine, "Indication", "Contraindication/Precautions")
        precautions = extract_section(medicine, "Contraindication/Precautions", "Dosage schedule")
        dosage_schedule = extract_section(medicine, "Dosage schedule", "Adverse effects")
        effects = extract_section(medicine, "Adverse effects", "Drug and food interaction")
        interactions = extract_section(medicine, "Drug and food interaction", "Patient information")
        patient_info = extract_section(medicine, "Patient information", "END")

        extracted_data.append([medicine_name, dosage_form, indication, precautions, 
                               dosage_schedule, effects, interactions, patient_info])
    return extracted_data

def save_to_csv(data, output_csv):
    """Saves extracted data to a CSV file."""
    headers = ["Medicine Name", "Dosage Form and Strength", "Indication", "Precautions", 
               "Dosage Schedule", "Effects", "Interactions", "Patient Information"]
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

# ===== RUN THE SCRIPT ===== #
pdf_path = "section1.pdf"  # Change to actual file path
output_csv = "section1_info.csv"

pdf_text = extract_text_from_pdf(pdf_path)
medicine_data = extract_medicine_info(pdf_text)
save_to_csv(medicine_data, output_csv)

print(f"Extraction complete! Data saved to {output_csv}")
