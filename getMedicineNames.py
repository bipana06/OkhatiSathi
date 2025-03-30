import fitz  # PyMuPDF
import re

def extract_pdf_data(pdf_path, pages):
    """
    Extract indexed section headings and medicine names separately from specified pages.

    :param pdf_path: Path to the PDF file
    :param pages: List of page numbers to extract from (1-based index)
    :return: (list of indexed headings, list of medicine names)
    """
    headings = []
    medicines = []

    # heading_pattern = re.compile(r"^\d+(\.\d+)*\s+[A-Za-z\-\s]+")  # Matches "X.Y Title" or "X.Y.Z Title"
    heading_pattern = re.compile(r"^\d+(\.\d+)*(\.\d+)*\s+[A-Za-z\-\s]+")  # Matches "X.Y Title", "X.Y.Z Title", or "X.Y.Z.A Title"
    medicine_pattern = re.compile(r"^[A-Za-z\-]+(?:\s+[A-Za-z\-]+)*$")  # Matches single/multi-word medicine names

    doc = fitz.open(pdf_path)
    for page_num in pages:
        if page_num - 1 < len(doc):  # Convert to 0-based index
            text = doc[page_num - 1].get_text("text")
            lines = text.split("\n")
            for line in lines:
                stripped_line = line.strip()
                if heading_pattern.match(stripped_line):
                    headings.append(stripped_line)
                elif medicine_pattern.match(stripped_line):
                    medicines.append(stripped_line)

    doc.close()
    return headings, medicines

# Example usage
pdf_file = "godFile.pdf"
pages_to_extract = [62, 63, 82, 83, 84, 116, 117, 138, 146, 162, 174, 175, 176, 177, 203, 230, 231, 252, 253, 254, 306, 307, 320, 321, 336, 352, 353, 382, 394, 395, 418, 419, 432, 433, 434]  # Specify page numbers

section_headings, medicine_names = extract_pdf_data(pdf_file, pages_to_extract)

# Now section_headings and medicine_names contain the extracted data as lists.
print("Section Headings:", section_headings)
print()
print("Medicine Names:", medicine_names)