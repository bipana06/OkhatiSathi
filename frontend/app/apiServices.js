const API_BASE_URL = 'http://127.0.0.1:8000'; 
// const API_BASE_URL = "https://okhatisathi.onrender.com"

// Function to fetch the audio for a given medicine name
export const getMedicineAudio = async (medicineName) => {
  try {
    console.log(`[API] Fetching audio for: ${medicineName}`);
    const response = await fetch(`${API_BASE_URL}/get_medicine_audio/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        medicine_name: medicineName,
      }),
    });

    console.log(`[API] Audio fetch response status: ${response.status}`);
    if (!response.ok) {
        let errorBody = 'Could not retrieve error details.';
        try {
            errorBody = await response.text();
        } catch (e) {
            console.error("Could not parse error response body:", e);
        }
        throw new Error(`Failed to fetch audio (${response.status} ${response.statusText}): ${errorBody}`);
    }

    // Handle the response as a Blob (binary data)
    const blob = await response.blob();
    console.log(`[API] Audio blob received, size: ${blob.size}, type: ${blob.type}`);

    if (blob.size === 0) {
        throw new Error("Received empty audio blob.");
    }

    // Create an object URL from the blob. This URL is temporary and tied to the current document.
    const audioUrl = URL.createObjectURL(blob);
    console.log(`[API] Created object URL for audio: ${audioUrl}`);

    return audioUrl; // Return the object URL
  } catch (error) {
    console.error('[API] Error fetching audio:', error);
    throw error;
  }
};


// Function to process the medicine image using base64 string
export const processMedicineImage = async (base64Image) => {
  try {
    console.log('[API] Sending image data to /process_medicine_image/');

    const response = await fetch(`${API_BASE_URL}/process_medicine_image/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      
      body: JSON.stringify({
        photo: base64Image, 
      }),
    });

    console.log(`[API] Process image response status: ${response.status}`);

    const jsonResponse = await response.json();

    if (!response.ok) {
      const errorMessage = jsonResponse.detail || `Server error (${response.status} ${response.statusText})`;
      console.error('[API] Error response from process image:', jsonResponse);
      throw new Error(errorMessage);
    }

    console.log('[API] Response from process medicine image:', jsonResponse);
    return jsonResponse; 

  } catch (error) {
    console.error('[API] Error in processMedicineImage:', error);
    throw error;
  }
};

export const getMedicineExplanationText = async (medicineName) => {
  try {
    console.log(`[API] Fetching explanation text for: ${medicineName}`);
    const response = await fetch(`${API_BASE_URL}/get_medicine_explanation_text/`, { // Ensure this matches your new endpoint
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        medicine_name: medicineName,
      }),
    });

    console.log(`[API] Explanation text response status: ${response.status}`);
    const jsonResponse = await response.json(); 

    if (!response.ok) {
      const errorMessage = jsonResponse.detail || `Server error (${response.status} ${response.statusText})`;
      console.error('[API] Error response fetching explanation text:', jsonResponse);
      throw new Error(errorMessage);
    }

    if (!jsonResponse.explanation_text) {
         console.error('[API] Explanation text missing from response:', jsonResponse);
         throw new Error("Explanation text not found in server response.");
    }

    console.log('[API] Received explanation text.');
    return jsonResponse.explanation_text; // Return the text string

  } catch (error) {
    console.error('[API] Error fetching explanation text:', error);
    throw error; 
  }
};


export default {
  getMedicineAudio,
  processMedicineImage,
};