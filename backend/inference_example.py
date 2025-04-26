import sys
import os

# Add the full absolute path to ./tacotron2 to sys.path
tacotron_path = os.path.abspath("tacotron2")
if tacotron_path not in sys.path:
    sys.path.insert(0, tacotron_path)

import nepalitts  


text_to_synthesize_1 = "कोमल घर गयो।"
output_directory = "audio_files"
output_file_1 = os.path.join(output_directory, "synthesized_audio_1.wav")
sampling_rate = 22050  # Set the sampling rate to 22050 Hz
print(f"Attempting to synthesize: '{text_to_synthesize_1}'")
# Call nepalitts.nepalitts, which will now return audio data (audio_numpy)
synthesized_audio = nepalitts.nepalitts(text=text_to_synthesize_1)


if synthesized_audio is not None:  # Check if the synthesis was successful
    print(f"Successfully synthesized audio.")
    # Now you can save the audio data to a file if you need to
    import scipy.io.wavfile as wavfile
    wavfile.write(output_file_1, sampling_rate, synthesized_audio)
    print(f"Audio saved to {output_file_1}")
else:
    print("Synthesis failed for the first text.")