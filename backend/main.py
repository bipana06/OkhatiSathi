from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import traceback
import json
import requests
from gtts import gTTS
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import HTTPException
from datetime import datetime
from pydantic import BaseModel
import base64
from OCR import perform_ocr  
from recognition import extract_medicine_name 
from dotenv import load_dotenv
import sys
# Add the full absolute path to ./tacotron2 to sys.path
tacotron_path = os.path.abspath("tacotron2")
if tacotron_path not in sys.path:
    sys.path.insert(0, tacotron_path)

import nepalitts  
# Create FastAPI app
app = FastAPI()

TEMP_IMAGE_DIR = "temp_images"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Medicine Data
filename = "MedicineDataset.json"

with open(filename, "r", encoding="utf-8") as file:
    medicine_data = json.load(file)

class MedicineResponse(BaseModel):
    status: str
    message: str
    drug_name: str | None = None
    matching_name: str | None = None
    synonyms: list[str] | None = None
    ocr_text: str | None = None
    audio_available: bool = False
    audio_file: str | None = None
    error: str | None = None


class ImagePayload(BaseModel):
    photo: str  # Base64-encoded image string

# Find Closest Medicine Name
def find_closest_medicine(match, synonyms,  data):
    medicine_names = [entry["Medicine"] for entry in data]
    
    # Check for exact match in medicine names
    for medicine in medicine_names:
        if match == medicine:
            print(f"Exact match found: {medicine}")
            return medicine

    # If no match found, check in synonyms
    for medicine in medicine_names:
        for synonym in synonyms:
            if synonym.upper() == medicine:
             print(f"Match found in synonyms: {medicine}")
             return medicine

    return None

# Extract Medicine Information
def get_medicine_info(medicine_name, data):
    for entry in data:
        if entry["Medicine"] == medicine_name:
            return entry
    return None

# Request Model
class MedicineRequest(BaseModel):
    medicine_name: str


@app.post("/process_medicine_image/", response_model=MedicineResponse)
async def process_medicine_image(payload: ImagePayload):
    """
    Receives a base64 encoded image, decodes it, performs OCR,
    identifies the medicine, and returns info.
    """
    temp_file_path = None  # Initialize path
    try:
        # Decode the base64 string
        try:
            
            missing_padding = len(payload.photo) % 4
            if missing_padding:
                payload.photo += '=' * (4 - missing_padding)
            image_data = base64.b64decode(payload.photo)
            print(f"Received and decoded image data, length: {len(image_data)} bytes")
        except base64.binascii.Error as decode_error:
            print(f"Base64 decoding error: {decode_error}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 string: {decode_error}")
        except Exception as e:
             print(f"Error during base64 decode: {e}")
             raise HTTPException(status_code=400, detail="Could not decode base64 image data.")

        # Save the image to a temporary file (use a unique name if handling concurrent requests)
        # For simplicity, using a fixed name here, but consider uuid or timestamp for production
        temp_file_path = os.path.join(TEMP_IMAGE_DIR, "uploaded_image_temp.jpg") # Use .jpg or detect format
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(image_data)
        print(f"Temporary image saved to: {temp_file_path}")

        # Perform OCR on the image
        try:
            ocr_text = perform_ocr(temp_file_path)
            print(f"OCR Result: '{ocr_text}'")
            if not ocr_text or ocr_text.strip() == "":
                 return MedicineResponse(
                    status="error",
                    message="OCR failed to extract any text from the image.",
                    audio_available=False
                 )
        except Exception as ocr_error:
             print(f"Error during OCR processing: {ocr_error}")
             
             # Return a generic error to the client
             return MedicineResponse(
                status="error",
                message=f"An error occurred during OCR: {ocr_error}",
                audio_available=False
             )


        # Extract medicine name from OCR text 
        drug_name, matching_name, synonyms = extract_medicine_name(ocr_text)
        print(f"Extracted Info: drug_name='{drug_name}', matching_name='{matching_name}', synonyms={synonyms}")


        # Handle OCR results
        if not drug_name or not matching_name:
            return MedicineResponse(
                status="success", # OCR worked, but no name found
                message="Medicine name not identified in the image text.",
                ocr_text=ocr_text,
                audio_available=False
            )

        # Find the medicine in the database
        medicine_in_database = find_closest_medicine(matching_name.upper(), synonyms, medicine_data)

        if not medicine_in_database:
            return MedicineResponse(
                status="success", # OCR worked, name identified, but not in DB
                message="Identified medicine name not found in our database.",
                matching_name=matching_name,
                synonyms=synonyms,
                ocr_text=ocr_text,
                audio_available=False
            )

        # indication that audio can be generated
        return MedicineResponse(
            status="success",
            message="Medicine identified. Audio can be generated.",
            drug_name=medicine_in_database, 
            matching_name=matching_name,
            synonyms=synonyms,
            ocr_text=ocr_text,
            audio_available=True
        )

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error processing image: {type(e).__name__}"
        )
    finally:
        # Clean up the temporary file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Removed temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                print(f"Error removing temporary file {temp_file_path}: {cleanup_error}")


@app.post("/get_medicine_audio/")
async def get_medicine_audio_endpoint(request: MedicineRequest):
    """Generates and returns the audio explanation for a given medicine name."""
    # Ensure the output directory exists
    AUDIO_OUTPUT_DIR = "audio_files"
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

    # Use uppercase for consistency if your data keys are uppercase
    medicine_name_upper = request.medicine_name.upper()
    print(f"Received request for audio for: {medicine_name_upper}")


    medicine_info = get_medicine_info(medicine_name_upper, medicine_data)
    if not medicine_info:
         print(f"Medicine '{medicine_name_upper}' not found in data for audio generation.")
         raise HTTPException(status_code=404, detail=f"Medicine '{request.medicine_name}' not found")

    medicine_text = json.dumps(medicine_info, indent=2)

    # Load environment variables from .env file
    load_dotenv()

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Access the API key from .env
    if not GROQ_API_KEY:
        print("GROQ_API_KEY not found in environment variables.")
        raise HTTPException(status_code=500, detail="Server configuration error: Missing Groq API Key")

    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

    async def get_layman_explanation(text: str) -> str:
        print("Requesting explanation from Groq...")
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

        payload = {
            "model": "llama3-70b-8192", # Check for current recommended model
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful Nepali nurse explaining medicine information in simple, clear Nepali language in devanagari script suitable for text-to-speech. Avoid medical jargon and complex sentence structures. Do not use markdown like * or :. Give everything in a small sentence (only 1 sentence)"
                },
                {
                    "role": "user",
                    "content": f"Please explain the following medicine information simply in Nepali: {text}"
                }
            ],
            "max_tokens": 500, # Allow longer explanation
            "temperature": 0.7 # Adjust creativity/factuality balance
        }
        try:
            response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30) # Add timeout
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            response_json = response.json()
            explanation = response_json["choices"][0]["message"]["content"]
            print("Received explanation from Groq.")
            return explanation
        
        except requests.exceptions.RequestException as req_err:
             print(f"Error calling Groq API: {req_err}")
             raise HTTPException(status_code=503, detail=f"Failed to get explanation from AI service: {req_err}")
        
        except (KeyError, IndexError) as parse_err:
             print(f"Error parsing Groq response: {parse_err}. Response: {response_json}")
             raise HTTPException(status_code=500, detail="Error parsing response from AI service")


    try:
        layman_text = await get_layman_explanation(medicine_text)
        print(f"Layman text (Nepali): {layman_text[:100]}...") # Log beginning of text
    except HTTPException as groq_http_exc:
         # Propagate specific errors from Groq call
         raise groq_http_exc
    except Exception as explanation_err:
         print(f"Error getting layman explanation: {explanation_err}")
         raise HTTPException(status_code=500, detail="Failed to generate explanation text.")

    mp3_filename = os.path.join(AUDIO_OUTPUT_DIR, "medicine_audio.mp3")

    try:
        print(f"Generating TTS audio, saving to: {mp3_filename}")
        
        
        # tts = gTTS(text=layman_text, lang='ne', slow=False) # lang='ne' for Nepali
        # tts.save(mp3_filename)
        
        output_directory = "audio_files"
        output_file_1 = mp3_filename
        sampling_rate = 22050  # Set the sampling rate to 22050 Hz
        # Call nepalitts.nepalitts, which will now return audio data (audio_numpy)
        synthesized_audio = nepalitts.nepalitts(text=layman_text)
        if synthesized_audio is not None:  # Check if the synthesis was successful
            print(f"Successfully synthesized audio.")
            # Now you can save the audio data to a file if you need to
            import scipy.io.wavfile as wavfile
            wavfile.write(output_file_1, sampling_rate, synthesized_audio)
            print(f"Audio saved to {output_file_1}")
        else:
            print("Synthesis failed for the first text.")

        print("TTS audio generated successfully.")
    except Exception as tts_error:
        print(f"Error generating TTS audio: {tts_error}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate audio file: {tts_error}")

    # Send the MP3 File as a Response
    if os.path.exists(mp3_filename):
        #  return FileResponse(mp3_filename, media_type="audio/mpeg", filename=os.path.basename(mp3_filename))
        return FileResponse(mp3_filename, media_type="audio/mpeg", filename=os.path.basename(mp3_filename))
    else:
         print(f"Generated MP3 file not found at path: {mp3_filename}")
         raise HTTPException(status_code=500, detail="Audio file was generated but could not be found.")

