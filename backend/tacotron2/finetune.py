# Cell 1: Imports and Initial Setup (No changes needed usually)
import torch
import os
import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Add tacotron2 directory and submodules to Python path (Ensure these paths are correct)
if os.getcwd().endswith('tacotron2'):
    sys.path.append('.') # If running notebook from tacotron2 dir
    # sys.path.append('waveglow') # Add waveglow submodule path if needed for inference later
    # sys.path.append('../hifi-gan') # Add hifi-gan path if needed for inference later
else:
    # Adjust these paths if your notebook is elsewhere
    # Example: Assuming tacotron2 folder is in the same directory as the notebook
    tacotron2_path = './tacotron2' # Adjust if needed
    if not os.path.isdir(tacotron2_path):
         raise FileNotFoundError(f"Tacotron2 directory not found at: {tacotron2_path}")
    sys.path.insert(0, tacotron2_path) # Add tacotron2 path first
    # Add hifi-gan path if needed for inference later
    # hifi_gan_path = '../hifi-gan' # Adjust relative path if needed
    # if os.path.isdir(hifi_gan_path):
    #     sys.path.append(hifi_gan_path)

print(f"Python Paths: {sys.path}")
try:
    print(f"Librosa version: {librosa.__version__}")
except NameError:
    print("Librosa might not be installed or imported correctly.")


# Cell 2: Define Paths (Adjust as needed)
output_directory_base = './outdir_sweep' # Base directory for all sweep runs
log_directory_name = 'logs'             # Subdirectory name for TensorBoard logs within each run's output dir
training_files = './filelists/train_list.txt' # Path to training list
validation_files = './filelists/val_list.txt' # Path to validation list
# Optional: Path to a pretrained model for warm start (usually None for sweeps)
# Set to None if you want to train from scratch for each sweep run
pretrained_checkpoint_path = "tacotron2_statedict.pt" # or "path/to/your/pretrained_tacotron2.pt"
# Optional: Pronunciation dictionary for inference later
# pronunciation_dict_file = './merged.dict.txt'

# Create base output directory
os.makedirs(output_directory_base, exist_ok=True)

print(f"Base Output Directory for Sweeps: {output_directory_base}")
print(f"Training Files: {training_files}")
print(f"Validation Files: {validation_files}")
print(f"Pretrained Checkpoint for Warm Start: {pretrained_checkpoint_path}")

# Cell 3: Updated HParams Class Definition
# Ensure text.symbols is available, might require importing specific symbols or text processing
try:
    from text import symbols
except ImportError:
    print("Warning: Could not import symbols from text module. Using dummy length.")
    # Define a placeholder if symbols cannot be imported - REPLACE this if needed
    class DummySymbols: # Simple placeholder if text module/symbols aren't found
        def __len__(self): return 100
    symbols = DummySymbols()

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
        # self.ignore_layers=['embedding.weight'] # Layers to ignore when loading warm start checkpoint

        ################################
        # Data Parameters             #
        ################################
        self.load_mel_from_disk=False
        self.training_files='./filelists/train_list.txt' # Will be overwritten by global path
        self.validation_files='./filelists/val_list.txt' # Will be overwritten by global path
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
# Cell 7: Training Execution Function for W&B Sweeps

import wandb
import traceback
# Ensure train function is imported from your modified train.py
# Ensure HParams class definition is available
try:
    from train import train
    print("Successfully imported train function from train.py")
except ImportError:
    print("ERROR: Could not import train function from train.py.")
    print("Ensure train.py is in the Python path and has been modified.")
    # Define a dummy train function to allow script execution without error,
    # but sweeps won't work. Replace this with actual fix.
    def train(*args, **kwargs):
        print("ERROR: Dummy train function called. Real function not imported!")
        raise ImportError("train function not found")

# --- Function to be called by W&B Agent ---
def run_training_sweep():
    run = None # Initialize run to None
    try:
        # Initialize W&B for this specific run
        # Config is automatically populated by the W&B agent
        run = wandb.init() # Project name is usually set by `wandb sweep` or agent command
        print(f"--- Starting W&B Run: {run.name} (ID: {run.id}) ---")

        # --- Load default HParams ---
        hparams = HParams() # Load your defaults defined in the class

        # --- OVERRIDE HParams with W&B Sweep Config ---
        config = wandb.config # Access hyperparameters for this run
        print("Sweep Configuration for this run:")
        for key, value in config.items():
            if hasattr(hparams, key):
                print(f"  Overriding hparams.{key}: {getattr(hparams, key)} -> {value}")
                setattr(hparams, key, value)
            else:
                print(f"  Warning: Hyperparameter '{key}' from sweep config not found in HParams class.")

        # --- Set Fixed Paths and Non-Tunable Parameters ---
        hparams.training_files = training_files
        hparams.validation_files = validation_files
        # Ensure distributed settings match the environment (usually False for single-agent runs)
        hparams.distributed_run = False # Override if agent controls multiple GPUs
        hparams.n_gpus = 1
        hparams.rank = 0

        # --- Create Unique Output Directory for this Run ---
        # Use W&B run ID for unique directory name
        run_output_directory = os.path.join(output_directory_base, run.id)
        os.makedirs(run_output_directory, exist_ok=True)
        print(f"Run-specific Output Directory: {run_output_directory}")

        # --- Set log directory within the run-specific output directory ---
        run_log_directory = os.path.join(run_output_directory, log_directory_name)
        # No need to create run_log_directory here, train.py's logger setup handles it

        # --- Select Checkpoint for Warm Start (Usually None for Sweeps) ---
        # If pretrained_checkpoint_path is set globally, use it. Otherwise None.
        current_checkpoint_path = pretrained_checkpoint_path
        use_warm_start = (current_checkpoint_path is not None)
        # Ensure ignore_layers is set correctly if warm starting
        if use_warm_start and not hasattr(hparams, 'ignore_layers'):
             hparams.ignore_layers = ['embedding.weight'] # Default ignore for warm start

        print("\n--- Final HParams for Training Run ---")
        for k, v in vars(hparams).items():
            print(f"  {k}: {v}")
        print("--------------------------------------\n")


        # --- Start Training (Call the modified function from train.py) ---
        print('Calling train function...')
        train(output_directory=run_output_directory,   # <<< Pass run-specific output dir
              log_directory=log_directory_name,       # <<< Pass log subdir name
              checkpoint_path=current_checkpoint_path,# Path for *loading* pretrained/resume model
              warm_start=use_warm_start,              # Use warm start if checkpoint provided
              n_gpus=hparams.n_gpus,                  # Number of GPUs
              rank=hparams.rank,                      # GPU rank
              group_name=hparams.group_name,          # Distributed group name
              hparams=hparams,                        # Pass the configured hparams
              use_wandb=True                          # <<< Enable W&B logging inside train >>>
             )
        print(f"--- W&B Run {run.name} Finished Successfully ---")

    except ImportError as e:
         print(f"\nERROR during run {run.id if run else 'unknown'}: Could not import necessary modules.")
         print(e)
         traceback.print_exc()
         # Optional: Log error to W&B if run initialized
         # if run: run.log({"error": str(e)})
         raise # Re-raise error to stop the agent if imports fail

    except Exception as e:
        print(f"\n--- ERROR during W&B Run {run.id if run else 'unknown'} ---")
        print(e)
        traceback.print_exc()
        # Optional: Log error to W&B if run initialized
        # if run: run.log({"error": str(e)})

    finally:
        # Ensure W&B run finishes properly
        if run:
            run.finish()
            print(f"--- W&B Run {run.name} Finalized ---")


# --- How to Run W&B Sweeps ---
# 1. Save this notebook as a Python script (e.g., `tacotron_sweep.py`)
#    OR ensure your environment can execute notebook cells via `wandb agent`.
#    Using a .py script is generally more robust for agents.
#
# 2. Create the `sweep.yaml` file (as defined in Cell 6).
#
# 3. Initialize the Sweep (run this in your terminal ONCE):
#    wandb sweep sweep.yaml
#    (This will output a SWEEP_ID like `USERNAME/PROJECT_NAME/SWEEP_ID`)
#
# 4. Run the W&B Agent (run this in your terminal, potentially multiple times):
#    wandb agent USERNAME/PROJECT_NAME/SWEEP_ID
#    (The agent will pick up jobs, execute this script which calls run_training_sweep)

# --- Optional: Single Test Run (without agent) ---
# To test the setup with one specific configuration without starting a full sweep:
def run_single_test():
     print("--- Running Single Test Run ---")
     # Define the config you want to test
     test_config = {
         "learning_rate": 1e-4,
         "weight_decay": 1e-6,
         "p_attention_dropout": 0.1,
         "p_decoder_dropout": 0.1,
         "batch_size": 32,
         # Add other parameters from your sweep config if needed
     }
     # Initialize W&B manually for the test run
     wandb.init(project="nepali-tts-hpc-test", config=test_config) # Use a test project
     run_training_sweep() # Call the main training function

# --- Main execution block for script ---
# This allows `wandb agent` to execute the script and call run_training_sweep
if __name__ == '__main__':
     # Check if a command-line argument '--test' is provided
     import argparse
     parser = argparse.ArgumentParser()
     parser.add_argument('--test', action='store_true', help='Run a single test configuration instead of waiting for sweep agent.')
     args, unknown = parser.parse_known_args() # Use parse_known_args to ignore agent args

     if args.test:
          run_single_test()
     else:
          # Default behavior for `wandb agent`: call the sweep function
          # The agent injects its config, so wandb.init() inside run_training_sweep works correctly.
          print("Script started by W&B Agent (or directly without --test). Running sweep function...")
          run_training_sweep()
    

