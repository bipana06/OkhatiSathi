
# This script is designed to be called for inference after training a Tacotron 2 model.

# Cell 1: Imports and Initial Setup (Minimal changes for path handling)
import torch
import os
import sys
import librosa
import numpy as np
from collections import OrderedDict # Added for handling state_dict prefix

PRINT_ENABLED = False  # Set to False to disable all prints
import builtins

real_print = print  # Keep the original print function

def print(*args, **kwargs):
    if PRINT_ENABLED:
        real_print(*args, **kwargs)

# --- Determine script's own directory ---
# This ensures paths are relative to the script, not the current working directory
try:
    # Use __file__ to get the script's path
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    print(f"Script directory: {script_dir}")
except NameError:
    # If __file__ is not defined (e.g., in interactive environments like Jupyter),
    # fallback to current working directory, but warn the user.
    script_dir = os.getcwd()
    print(f"Warning: __file__ not defined. Using current working directory as script directory: {script_dir}")
    print("Calling this script from a different directory might cause issues.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Add required directories to Python path based on script location ---
# Assuming tacotron2 and hifi-gan are subdirectories relative to the script
tacotron2_path = os.path.join(script_dir, 'tacotron2')
hifi_gan_repo_path = os.path.join(script_dir, 'hifi-gan')

# Add tacotron2 directory first
if os.path.isdir(tacotron2_path):
    if tacotron2_path not in sys.path:
        sys.path.insert(0, tacotron2_path)
        print(f"Added to sys.path: {tacotron2_path}")
else:
    print(f"Warning: Tacotron2 directory not found at: {tacotron2_path}")
    print("Some modules might not be importable.")

# Add hifi-gan directory
if os.path.isdir(hifi_gan_repo_path):
     if hifi_gan_repo_path not in sys.path:
        sys.path.append(hifi_gan_repo_path)
        print(f"Added to sys.path: {hifi_gan_repo_path}")
else:
    print(f"Warning: HiFi-GAN directory not found at: {hifi_gan_repo_path}")
    print("HiFi-GAN modules might not be importable.")


# Optional: Add current script directory to path if needed for local imports
# if script_dir not in sys.path:
#      sys.path.append(script_dir)
#      print(f"Added to sys.path: {script_dir}")

print(f"Python Paths: {sys.path}")
try:
    print(f"Librosa version: {librosa.__version__}")
except NameError:
    print("Librosa might not be installed or imported correctly.")


# Cell 3: Updated HParams Class Definition
# Ensure text.symbols is available, might require importing specific symbols or text processing
# Define a placeholder if symbols cannot be imported - REPLACE this if needed
class DummySymbols: # Simple placeholder if text module/symbols aren't found
    def __len__(self): return 100
symbols = DummySymbols()
print(f"Dummy symbols length: {len(symbols)}")
try:
    from text import symbols
    print(f"Symbols loaded successfully. Number of symbols: {len(symbols)}")
except ImportError:
    print("Import failed: Could not import symbols from text module.")
    print(f"Sys path: {sys.path}")
    import traceback
    traceback.print_exc()
    print("Warning: Using dummy length for symbols.")
    


# You might need to import the HParams class definition if it's in a separate file,
# e.g., from hparams import HParams
# If it's not, define it here:
class HParams:
    def __init__(self) -> None:
        ################################
        # Experiment Parameters        #
        ################################
        self.epochs=200 # Max epochs per run (early stopping will likely trigger earlier)
        self.seed=1234
        self.distributed_run=False # Set True if using multi-GPU via launch script
        self.n_gpus=1 # Number of GPUs for distributed run
        self.rank=0 # Rank for distributed run
        self.group_name="group_name" # Group name for distributed run
        self.cudnn_enabled=True
        self.cudnn_benchmark=False # Set True if input size is constant, False otherwise
        self.fp16_run=False # Enable mixed precision (requires torch.cuda.amp)
        self.ignore_layers=['embedding.weight'] # Layers to ignore when loading warm start checkpoint

        ################################
        # Data Parameters             #
        ################################
        self.load_mel_from_disk=False
        # !!! These training/validation files paths are likely relative to the original
        # training script's location. You might need to make these absolute or
        # handle them differently if you intend to train from this script.
        # For inference, these specific paths might not be critical.
        self.training_files='./filelists/train_list.txt'
        self.validation_files='./filelists/val_list.txt'
        self.text_cleaners=['transliteration_cleaners'] # Adjust based on your data/language
        self.num_workers=4 # Number of workers for DataLoader (tune based on system)
        self.pin_memory=True # Set True if using GPU, helps speed up data transfer

        ################################
        # Audio Parameters             #
        ################################
        self.max_wav_value=32768.0
        self.sampling_rate=22050
        self.filter_length=1024
        self.hop_length=256
        self.win_length=1024
        self.n_mel_channels=80
        self.mel_fmin=0.0
        self.mel_fmax=8000.0

        ################################
        # Model Parameters             #
        ################################
        self.n_symbols=len(symbols)
        self.symbols_embedding_dim=512
        # Encoder parameters
        self.encoder_kernel_size=5
        self.encoder_n_convolutions=3
        self.encoder_embedding_dim=512
        # Decoder parameters
        self.n_frames_per_step=1  # currently only 1 is supported
        self.decoder_rnn_dim=1024
        self.prenet_dim=256
        self.max_decoder_steps=1000 # Max steps during inference generation
        self.gate_threshold=0.5 # Threshold for stop token during inference
        self.p_attention_dropout=0.1 # <<< TUNABLE regularization >>>
        self.p_decoder_dropout=0.1   # <<< TUNABLE regularization >>>
        # Attention parameters
        self.attention_rnn_dim=1024
        self.attention_dim=128
        # Location Layer parameters
        self.attention_location_n_filters=32
        self.attention_location_kernel_size=31
        # Mel-post processing network parameters
        self.postnet_embedding_dim=512
        self.postnet_kernel_size=5
        self.postnet_n_convolutions=5

        ################################
        # Optimization Hyperparameters #
        ################################
        self.use_saved_learning_rate=False # Set True only when resuming and wanting the exact LR from checkpoint
        self.learning_rate=1e-4       # <<< TUNABLE hyperparameter >>>
        self.weight_decay=1e-6        # <<< TUNABLE regularization >>>
        self.grad_clip_thresh=1.0
        self.batch_size=32             # <<< TUNABLE (adjust based on GPU memory) >>>
        self.mask_padding=True  # set model's padded outputs to padded values

        # --- Training Control Parameters ---
        self.log_interval = 100 # Log training progress every N iterations
        self.iters_per_checkpoint = 1000 # Perform validation & checkpointing every N iterations

        # --- Early Stopping Parameters ---
        self.early_stopping_patience = 10 # How many validation checks without improvement before stopping (e.g., 10 * iters_per_checkpoint iterations)
        self.min_val_loss_delta = 0.0001 # Minimum improvement required in validation loss to reset patience

# Instantiate HParams once to have a base object (optional, as run_training_sweep creates its own)
# hparams_default = HParams()
# print("Default HParams created (will be overridden by sweep config).")

# Cell 8: Inference Setup (Keep this separate, run *after* training)
# --- Load the BEST checkpoint identified by your sweep runs ---
# You'll need to look at your W&B project dashboard to find the run ID
# and iteration number of the best performing model based on validation loss.

import json
import gdown, traceback
# Make sure model, layers, text utils etc. are imported
# These should now be importable because tacotron2_path is in sys.path
try:
    from model import Tacotron2
    from layers import TacotronSTFT
    from text import text_to_sequence
    print("Successfully imported Tacotron 2 modules.")
except ImportError:
    print("Import failed: Could not import Tacotron 2 modules (model, layers, text).")
    print(f"Sys path: {sys.path}")
    import traceback
    traceback.print_exc()
    # Define dummy classes if needed to avoid NameErrors later
    class Tacotron2: pass
    class TacotronSTFT: pass
    def text_to_sequence(*args, **kwargs): raise NotImplementedError("text_to_sequence not imported")


# --- HiFi-GAN specific imports ---
try:
    # hifi_gan_repo_path is already added to sys.path above
    from env import AttrDict
    from meldataset import MAX_WAV_VALUE
    from models import Generator as HiFiGAN_Generator
    print("Successfully imported HiFi-GAN modules.")
except ImportError:
    print("Import failed: Could not import HiFi-GAN components.")
    print(f"Sys path: {sys.path}")
    import traceback
    traceback.print_exc()
    # Define dummy classes if needed to avoid NameErrors later
    class AttrDict(dict): pass
    class HiFiGAN_Generator: pass
    MAX_WAV_VALUE = 32767.0
    print("Warning: Could not import HiFi-GAN components. Ensure HiFi-GAN repo is in Python path.")


# --- Configuration (Paths adjusted to be relative to the script) ---
# <<< IMPORTANT: Update this path after finding your best sweep run >>>
# Example: best_run_id = "your_wandb_run_id" # Find this in W&B dashboard
# Example: best_checkpoint_iteration = 15000 # Find the best iteration number
# best_checkpoint_filename = f"checkpoint_best_iter_{best_checkpoint_iteration}_loss_XXXX.pt" # Check exact filename
# tacotron2_checkpoint_path = os.path.join(output_directory_base, best_run_id, best_checkpoint_filename)
# print(f"Attempting to load best checkpoint: {tacotron2_checkpoint_path}")

# --- Placeholder path - ADJUST THIS to your actual checkpoint location relative to script_dir ---
# Example: Assuming checkpoint is in a 'models' subdirectory next to the script
# tacotron2_checkpoint_path = os.path.join(script_dir, 'models', 'checkpoint.pt')
# If it's in './final_checkpoint/checkpoint.pt' relative to the script dir:
tacotron2_checkpoint_path = os.path.join(script_dir, "final_checkpoint", "checkpoint.pt")
print(f"Tacotron 2 checkpoint path: {tacotron2_checkpoint_path}")


# --- HiFi-GAN Setup (Paths adjusted) ---
# hifigan_repo_path is already defined and added to sys.path
hifigan_config_path = os.path.join(hifi_gan_repo_path, 'config_v1.json')
hifigan_model_id = "1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW" # Universal model GDrive ID
# Save downloaded HiFi-GAN model relative to the script dir
hifigan_local_path = os.path.join(script_dir, 'hifigan_universal.pt') # Where to save downloaded HiFi-GAN

print(f"HiFi-GAN config path: {hifigan_config_path}")
print(f"HiFi-GAN local model path: {hifigan_local_path}")


# --- Download HiFi-GAN if needed ---
if not os.path.exists(hifigan_local_path):
    print("Downloading HiFi-GAN model...")
    try:
        
        gdown.download(f'https://drive.google.com/uc?id={hifigan_model_id}', hifigan_local_path, quiet=False)
        print("HiFi-GAN Download complete.")
    except Exception as e:
        print(f"Failed to download HiFi-GAN: {e}")
else:
    print("HiFi-GAN model already downloaded.")

# --- Load HiFi-GAN ---
hifigan_generator = None
if os.path.exists(hifigan_local_path) and os.path.exists(hifigan_config_path):
    print("Loading HiFi-GAN...")
    try:
        with open(hifigan_config_path) as f:
            hifigan_h = AttrDict(json.load(f))
        hifigan_generator = HiFiGAN_Generator(hifigan_h).to(device)
        state_dict_g = torch.load(hifigan_local_path, map_location=device)
        hifigan_generator.load_state_dict(state_dict_g['generator'])
        hifigan_generator.eval()
        hifigan_generator.remove_weight_norm()
        print("HiFi-GAN Loaded.")
    except Exception as e:
        print(f"Failed to load HiFi-GAN: {e}")
        traceback.print_exc()
        hifigan_generator = None # Ensure it's None if loading failed
else:
    print("HiFi-GAN model or config file not found. Skipping HiFi-GAN loading.")


# --- Load Trained Tacotron 2 ---
tacotron_model = None
hparams_inf = None # Define hparams_inf here so it's available later even if model loading fails
if os.path.exists(tacotron2_checkpoint_path):
    print(f"Loading Tacotron 2 checkpoint: {tacotron2_checkpoint_path}")
    try:
        # --- Load HParams used for the *best* run ---
        # Checkpoints saved by modified train.py should contain hparams
        checkpoint_dict = torch.load(tacotron2_checkpoint_path, map_location='cpu')
        if 'hparams' in checkpoint_dict:
             print("Loading HParams from checkpoint...")
             # Recreate HParams object from the dictionary saved in the checkpoint
             hparams_inf = HParams() # Start with default
             # Convert Namespace back to dict if necessary (older checkpoints)
             hparams_dict_from_checkpoint = checkpoint_dict['hparams']
             if not isinstance(hparams_dict_from_checkpoint, dict):
                 # Attempt to convert from Namespace or other object
                 try:
                     hparams_dict_from_checkpoint = vars(hparams_dict_from_checkpoint)
                 except TypeError:
                     print("Warning: Could not convert checkpoint hparams object to dict. Using default HParams for inference.")
                     hparams_dict_from_checkpoint = {} # Use empty dict to fallback to defaults

             # Update the default hparams object
             for k, v in hparams_dict_from_checkpoint.items():
                 if hasattr(hparams_inf, k): # Only update if the attribute exists in the HParams class
                     setattr(hparams_inf, k, v)
                 # else:
                 #      print(f"Warning: Checkpoint hparams key '{k}' not found in HParams class.")
             print("HParams loaded successfully from checkpoint.")
        else:
             print("Warning: HParams not found in checkpoint. Using default HParams for inference.")
             # Fallback to default HParams object if not found
             hparams_inf = HParams() # Load default HParams

        # --- Configure HParams for Inference ---
        # These might differ from training settings
        hparams_inf.max_decoder_steps = 2000 # Increase max steps for potentially longer sentences
        hparams_inf.gate_threshold = 0.5    # Stop threshold (can be tuned)
        # Ensure batch size is 1 for inference
        hparams_inf.batch_size = 1
        # Ensure distributed run is off for inference
        hparams_inf.distributed_run = False
        # Ensure the n_symbols matches the loaded symbols length if using default HParams
        # If hparams were loaded from checkpoint, this should already match
        if isinstance(symbols, DummySymbols) and hparams_inf.n_symbols != len(symbols):
             print(f"Warning: HParams n_symbols ({hparams_inf.n_symbols}) does not match dummy symbols length ({len(symbols)}). Using dummy length.")
             hparams_inf.n_symbols = len(symbols)


        # --- Load Model ---
        # Tacotron2 class should be importable now
        from model import Tacotron2
        tacotron_model = Tacotron2(hparams_inf)

        # Load state dict (ensure keys match - handled by load_checkpoint in train.py during saving)
        state_dict = checkpoint_dict['state_dict']
        # Handle 'module.' prefix if the loaded checkpoint has it but inference model doesn't
        # or vice versa, though typically inference model doesn't have the prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k # Keep original key by default
            if k.startswith('module.') and not list(tacotron_model.state_dict().keys())[0].startswith('module.'):
                 # Checkpoint has 'module.', model doesn't
                 name = k[7:]
            elif not k.startswith('module.') and list(tacotron_model.state_dict().keys())[0].startswith('module.'):
                 # Checkpoint doesn't have 'module.', model does
                 name = 'module.' + k

            # Add the key if it exists in the model's state_dict
            if name in tacotron_model.state_dict():
                new_state_dict[name] = v
            # else:
            #      print(f"Warning: Skipping checkpoint key '{k}' (maps to '{name}') as it is not in the model's state_dict.")

        # Load the potentially modified state dictionary
        # strict=False allows loading even if some keys are missing (e.g., optimizer state)
        tacotron_model.load_state_dict(new_state_dict, strict=False)

        tacotron_model = tacotron_model.to(device).eval()
        # If using fp16 during training, you might use .half(), otherwise keep .float()
        # if hparams_inf.fp16_run: tacotron_model = tacotron_model.half()
        print("Tacotron 2 Loaded for Inference.")

    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at the specified path: {tacotron2_checkpoint_path}")
        print("Please update the path to your best trained model checkpoint.")
        tacotron_model = None # Ensure model is None if loading failed
        hparams_inf = HParams() # Load default hparams even if model loading failed
    except Exception as e:
        print(f"Failed to load Tacotron 2 model: {e}")
        traceback.print_exc()
        tacotron_model = None # Ensure model is None if loading failed
        hparams_inf = HParams() # Load default hparams even if model loading failed
else:
     print(f"ERROR: Tacotron 2 checkpoint path does not exist: {tacotron2_checkpoint_path}")
     print("Please train a model first or correct the path.")
     hparams_inf = HParams() # Load default hparams if checkpoint not found


# --- Inference Wrapper Setup ---
# These stay in your current file
tacotron_model_inf = tacotron_model
hifigan_model_inf = hifigan_generator
# hparams_inf is defined and potentially loaded from checkpoint above
device_inf = device
hparams_inf = hparams_inf # Ensure hparams_inf is available globally
import numpy as np
import torch
import scipy.io.wavfile as wavfile

def infer(text, output_filename=None):
    # Check if models and hparams_inf are loaded
    if tacotron_model_inf is None or hifigan_model_inf is None or hparams_inf is None:
        print("ERROR: Required models or HParams not loaded. Cannot perform inference.")
        # Attempt to load default HParams if they failed earlier
        if hparams_inf is None:
             print("Attempting to load default HParams...")
             try:

                 print("Default HParams loaded.")
             except Exception as hp_e:
                 print(f"Failed to load default HParams: {hp_e}")
                 return None # Cannot proceed without HParams

        # Even if default HParams loaded, if models are None, we can't proceed
        if tacotron_model_inf is None or hifigan_model_inf is None:
            return None


    print(f"\nInput Text: {text}")
    # Ensure text ends with a punctuation mark for Tacotron 2's stopping mechanism
    # Check if the last character is already a common punctuation mark (., !, ?)
    if not text or text[-1] not in ['.', '!', '?', ';']:
        processed_text = text + "." # Add a period if no punctuation
    else:
        processed_text = text # Use text as is

    try:
        # text_to_sequence should be importable because tacotron2_path is in sys.path
        from text import text_to_sequence
        sequence = np.array(text_to_sequence(processed_text, hparams_inf.text_cleaners))[None, :]
        sequence = torch.from_numpy(sequence).to(device_inf).long()
    except NameError:
         print("ERROR: text_to_sequence function not found. Ensure tacotron2 modules are loaded.")
         return None
    except Exception as e:
        print(f"Error during text_to_sequence: {e}")
        traceback.print_exc()
        return None

    with torch.no_grad():
        try:
            # Tacotron 2 inference
            mel_outputs, mel_outputs_postnet, _, alignments = tacotron_model_inf.inference(sequence)
        except Exception as e:
            print(f"Error during Tacotron 2 inference: {e}")
            traceback.print_exc()
            return None

        try:
            # HiFi-GAN inference
            y_g_hat = hifigan_model_inf(mel_outputs_postnet.float())
            audio = y_g_hat.squeeze() * MAX_WAV_VALUE
            audio_numpy = audio.cpu().numpy().astype('int16')
        except Exception as e:
            print(f"Error during HiFi-GAN inference: {e}")
            traceback.print_exc()
            return None

        if output_filename:  # If an output filename is provided, save the audio file
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created output directory: {output_dir}")

            wavfile.write(output_filename, hparams_inf.sampling_rate, audio_numpy)
            print(f"Saved audio to {output_filename}")

        # Return the audio data (NumPy array)
        return audio_numpy


# Example of how to use the infer function:
# audio_data = infer("नमस्ते संसार!")
# if audio_data is not None:
#     print(f"Generated audio data with shape: {audio_data.shape}")
#     # You can further process or play audio_data here

# Example saving to file:
# audio_data = infer("यो एउटा परीक्षण वाक्य हो।", output_filename="output_audio.wav")
# if audio_data is not None:
#      print("Inference complete and audio saved.")

# Assign the infer function to nepalitts name
nepalitts = infer