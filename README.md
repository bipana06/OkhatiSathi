# OkhatiSathi

OkhatiSathi is a medicinal assistant app designed to help users identify medicine names and provide detailed information about them in simple Nepali sentences. The app explains the medicine's purpose, dosage, risks, compatibility with other substances, and more, making it accessible to the general public.

## Features
- **Medicine Identification**: Uses OCR (Optical Character Recognition) powered by `pytesseract` to extract medicine names from images.
- **Drug Information Retrieval**: Employs drug named entity recognition to fetch relevant details about the identified medicine.
- **Text-to-Speech (TTS)**: Converts the retrieved information into spoken Nepali for ease of understanding.

## Data Sources
- **Medicine Data**: Collected, cleaned, and organized from the publicly available pdf regarding the medicines used in Nepal provided by the National Health Infirmary in Nepal. 
- **Nepali TTS Model**: Trained using publicly available Nepali audio and text datasets.

## Installation and Usage
Follow these steps to set up and run the app:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/bipana06/OkhatiSathi.git
    cd OkhatiSathi
    ```
2. **Backend Setup**:

    1. navigate to backend
    ```bash
    cd backend
    ```
    2. Virtual Environment 
    ```bash
    python -m venv campusenv
    source campusenv/bin/activate  # On Windows: campusenv\Scripts\activate
    ```
    3. Ensure you have Python installed. Then, run:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

    4. Add a .env file and add your ```GROQ_API_KEY```
    
3. **Run the Application**:
    ```bash
    cd frontend
    npm install
    npx expo start
    ```


4. **Input**:
    - Upload an image of the medicine.
    - The app will process the input and provide detailed information in Nepali.

5. **Output**:

    - Audio output in Nepali using the TTS model.

## Example Usage
1. Upload an image of a medicine label.
2. The app identifies the medicine name using OCR.
3. It retrieves information such as:
    - Purpose of the medicine.
    - Dosage instructions.
    - Risks and side effects.
    - Compatibility with food, drinks, or other medicines.
4. The app reads out the information in Nepali.

## Contributors
- **Bipana Bastola**
- **Manoj Dhakal**

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- National Health Infirmary, Nepal, for providing the medicine data.
- Open-source contributors for OCR, NER, and TTS tools.
- Publicly available Nepali datasets for training the TTS model.
