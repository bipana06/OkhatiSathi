# OkhatiSathi

OkhatiSathi is a medicinal assistant app designed to help users identify medicine names and provide detailed information about them in simple Nepali sentences. The app explains a medicine’s purpose, dosage, risks, compatibility, and more – making it accessible to the general public. It also includes a text-to-speech (TTS) feature that reads out the information in Nepali.

## Features

- **Medicine Identification**  
  Uses OCR (powered by `pytesseract`) to extract medicine names from images.

- **Drug Information Retrieval**  
  Employs drug named entity recognition (NER) to fetch details about the identified medicine.

- **Text-to-Speech (TTS) in Nepali**  
  Converts the retrieved information into spoken Nepali using a custom-trained TTS model (Tacotron2 based).

## Project Structure

```
OkhatiSathi/
├── backend/             # Contains the Python backend (OCR, NER, TTS, Tacotron2 inference)
│   ├── tacotron2/       # Tacotron2 model code (make sure that 'tacotron2/text/' is complete)
│   ├── requirements.txt
│   ├── .env             # Contains sensitive info like GROQ_API_KEY (create this file)
│   └── [other backend files]
├── frontend/            # Expo app for user interaction
│   ├── app/             # Expo Router based code (React Native)
│   ├── package.json
│   └── [other frontend files]
└── README.md
```

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/bipana06/OkhatiSathi.git
cd OkhatiSathi
```

### 2. Backend Setup

#### a. Create and Activate a Virtual Environment
Navigate to the backend folder and create a virtual environment. (Requires Python 3.8+)
```bash
cd backend
python -m venv campusenv
source campusenv/bin/activate    # On Windows: campusenv\Scripts\activate
```

#### b. Install Python Dependencies
Ensure that you have the required packages installed:
```bash
pip install -r requirements.txt
```

#### c. Configure Environment Variables
Create a `.env` file in the `backend` directory and add your API keys. For example:
```
GROQ_API_KEY=your_groq_api_key_here
```

> **Note:**  
> Ensure that your Tacotron2 directory contains the `text` subdirectory (with an `__init__.py` and `text_processing.py`) so that the model loads the correct symbols. This is required to match the checkpoint dimensions.

#### d. Start the Backend Server
Depending on your application entry point (for example, if it’s named `server.py` or similar), start the backend server:
```bash
python server.py
```
Check the console for messages such as “Tacotron2 model initialized.” and verify that no checkpoint loading errors occur.

### 3. Frontend Setup

#### a. Install Node Dependencies
Navigate to the `frontend` folder and install dependencies via npm:
```bash
cd ../frontend
npm install
```

#### b. Start the Expo Application
Start the Expo development server using npx:
```bash
npx expo start
```
Follow the instructions to run the app on a device, emulator, or via the web.

## Usage

1. **Input Image Upload:**  
   Open the frontend app and upload an image of a medicine label. The OCR will extract and identify the medicine.

2. **Information Retrieval:**  
   The backend will fetch additional medicine details and generate a simple Nepali explanation using the NER and TTS modules.

3. **Audio Output:**  
   The TTS system converts the text explanation into speech. You can listen to the generated audio, pause or stop it if needed (the frontend includes controls such as a “Stop Audio” button).

## Troubleshooting

- **Tacotron2 Checkpoint Errors:**  
  If you see errors regarding size mismatches in the embedding layer (e.g., “size mismatch for embedding.weight”), ensure that the `tacotron2/text` folder is complete and that the symbols list matches the one used during model training.

- **Dependencies and Environment Issues:**  
  - Verify the Python environment is activated when running the backend.
  - For any Expo-related problems, clear cache with:
    ```bash
    npx expo start --clear
    ```

- **API Keys and Environment Variables:**  
  Ensure your `.env` file contains the correct values (e.g., `GROQ_API_KEY`) so that external API calls function properly.

## Data Sources

- **Medicine Data:**  
  Collected and organized from public PDFs provided by the National Health Infirmary in Nepal.

- **Nepali TTS Model:**  
  Trained using publicly available Nepali audio and text datasets.

## Contributors

- **Bipana Bastola**
- **Manoj Dhakal**

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- National Health Infirmary, Nepal for providing medicine data.
- Open-source projects for OCR, NER, and TTS tools.
- Public Nepali datasets for training purposes.
