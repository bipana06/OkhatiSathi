
import os
import time
import argparse
import math
from numpy import finfo
import numpy as np # <<< CHANGE: Added for float('inf') >>>

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams # <<< CHANGE: Assuming HParams class is not defined here directly >>>
# If HParams class IS defined in this file, you don't need create_hparams usually
from collections import OrderedDict

import wandb # <<< Already imported, ensure it's used conditionally

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM) # <<< CHANGE: Use newer enum if available, otherwise keep dist.reduce_op.SUM >>>
    rt /= n_gpus
    return rt

# --- init_distributed, prepare_dataloaders, prepare_directories_and_logger, load_model ---
# --- warm_start_model, load_checkpoint (using your improved version) ---
# --- No changes needed in these functions above ---

def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)
    print("Done initializing distributed")

def prepare_dataloaders(hparams):
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)
    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(trainset, num_workers=getattr(hparams, 'num_workers', 1), shuffle=shuffle, # <<< CHANGE: Use hparam for num_workers if available >>>
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=getattr(hparams, 'pin_memory', False), # <<< CHANGE: Use hparam for pin_memory if available >>>
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn

def prepare_directories_and_logger(output_directory, log_directory, rank, hparams=None): # <<< CHANGE: Added hparams >>>
    logger = None
    if rank == 0:
        if not os.path.isdir(output_directory):
            print(f"Creating output directory: {output_directory}")
            os.makedirs(output_directory, exist_ok=True) # <<< CHANGE: Use exist_ok=True >>>
            # os.chmod(output_directory, 0o775) # Might cause issues on some systems
        log_path = os.path.join(output_directory, log_directory)
        print(f"TensorBoard Log directory: {log_path}")
        logger = Tacotron2Logger(log_path) # <<< CHANGE: Pass hparams to logger if it accepts them >>>
    return logger

def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if getattr(hparams, 'fp16_run', False): # <<< CHANGE: Safer access >>>
        # Ensure attention score mask uses fp16 min if necessary
        if hasattr(model.decoder.attention_layer, 'score_mask_value'):
             model.decoder.attention_layer.score_mask_value = finfo('float16').min
    # Distributed is handled later with DDP or apply_gradient_allreduce wrapper
    return model

def warm_start_model(checkpoint_path, model, ignore_layers):
    # Keep your existing warm_start_model
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if ignore_layers and len(ignore_layers) > 0: # <<< CHANGE: Check if ignore_layers is not None/empty >>>
        model_dict_clean = {}
        ignored_count = 0
        original_keys = set(model_dict.keys())
        model_state_keys = set(model.state_dict().keys())

        # Handle 'module.' prefix if necessary
        has_module_prefix = any(k.startswith('module.') for k in original_keys)
        model_needs_prefix = any(k.startswith('module.') for k in model_state_keys)

        for k, v in model_dict.items():
            original_k = k
            # Strip 'module.' prefix from checkpoint if model doesn't have it
            if has_module_prefix and not model_needs_prefix and k.startswith('module.'):
                k = k[7:]
            # Add 'module.' prefix if model needs it and checkpoint doesn't have it
            elif not has_module_prefix and model_needs_prefix and not k.startswith('module.'):
                 k = 'module.' + k

            # Check against ignore list
            should_ignore = False
            for pattern in ignore_layers:
                if k.startswith(pattern):
                    should_ignore = True
                    break
            if not should_ignore:
                 # Check if the key exists in the current model state dict before adding
                 if k in model_state_keys:
                      model_dict_clean[k] = v
                 else:
                      print(f"    Warning: Key '{k}' (original: '{original_k}') from checkpoint not found in model state_dict. Skipping.")
                      ignored_count += 1
            else:
                # print(f"    Ignoring layer: {k} (original: {original_k})") # Optional: verbose ignore log
                ignored_count += 1

        # Load the filtered state dict
        print(f"  Applying warm start: Ignored {ignored_count} keys based on ignore_layers. Loading {len(model_dict_clean)} keys.")
        # It's safer to load into a dummy dict first if using ignore_layers with warm start
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict_clean) # Update with loaded weights, keeping others initialized
        model.load_state_dict(dummy_dict)

    else: # Load all weights if ignore_layers is empty
        print("  Applying warm start: Loading all keys (ignore_layers is empty).")
        model.load_state_dict(model_dict) # Consider adding strict=False if needed

    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    # <<< Keep your improved load_checkpoint function here >>>
    assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    # Handle potential 'module.' prefix in saved state_dict
    saved_state_dict = checkpoint_dict['state_dict']
    model_needs_module_prefix = isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))

    if any(key.startswith('module.') for key in saved_state_dict.keys()) and not model_needs_module_prefix:
        print("Removing 'module.' prefix from checkpoint keys...")
        new_state_dict = OrderedDict()
        for k, v in saved_state_dict.items():
            if k.startswith('module.'): name = k[7:]
            else: name = k
            new_state_dict[name] = v
        model_state_dict_to_load = new_state_dict
    elif not any(key.startswith('module.') for key in saved_state_dict.keys()) and model_needs_module_prefix:
         print("Adding 'module.' prefix to checkpoint keys...")
         new_state_dict = OrderedDict()
         for k, v in saved_state_dict.items():
              name = 'module.' + k
              new_state_dict[name] = v
         model_state_dict_to_load = new_state_dict
    else:
        model_state_dict_to_load = saved_state_dict

    # Load model state, allowing missing keys for flexibility (e.g., different heads)
    # Set strict=False if warm starting certain parts or if architecture might differ slightly.
    # Set strict=True if resuming exact training state. For general loading, False might be safer.
    model.load_state_dict(model_state_dict_to_load, strict=False)
    print("Model state loaded.")

    learning_rate = None
    iteration = 0

    if optimizer is not None and 'optimizer' in checkpoint_dict:
        try:
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
            print("Optimizer state loaded.")
        except ValueError as e:
            print(f"Warning: Could not load optimizer state dict: {e}. Starting optimizer from scratch.")
    elif optimizer is not None:
        print("Warning: Optimizer state not found in checkpoint. Starting optimizer from scratch.")

    if 'learning_rate' in checkpoint_dict:
        learning_rate = checkpoint_dict['learning_rate']
        print(f"Learning rate ({learning_rate}) loaded from checkpoint.")
    else:
        print("Warning: Learning rate not found in checkpoint.")

    if 'iteration' in checkpoint_dict:
        iteration = checkpoint_dict['iteration']
    else:
        print("Warning: Iteration count not found in checkpoint. Starting from iteration 0.")

    # <<< CHANGE: Potentially load early stopping state if saved >>>
    # best_val_loss = checkpoint_dict.get('best_val_loss', float('inf'))
    # print(f"Best validation loss state loaded: {best_val_loss}")

    print("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration #, best_val_loss


# <<< CHANGE: Modified save_checkpoint to include optional early stopping state >>>
def save_checkpoint(model, optimizer, learning_rate, iteration, filepath, hparams=None, best_val_loss=None):
    """Saves checkpoint to disk."""
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    # Handle DataParallel wrapper by saving module state dict
    state_dict = model.state_dict()
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
         state_dict = model.module.state_dict()

    save_dict = {
        'iteration': iteration,
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate
    }
    # Optionally include hparams and best_val_loss in the checkpoint
    if hparams:
        save_dict['hparams'] = vars(hparams) if not isinstance(hparams, dict) else hparams
    if best_val_loss is not None:
        save_dict['best_val_loss'] = best_val_loss

    torch.save(save_dict, filepath)
    print(f"Checkpoint saved: {filepath}")


# <<< CHANGE: Modified validate to return loss >>>
def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, distributed_run, rank, hparams): # <<< CHANGE: Added hparams >>>
    """Handles all the validation scoring and calculation"""
    model.eval()
    with torch.no_grad():
        # Ensure val_sampler uses correct DistributedSampler arguments if needed
        val_sampler = DistributedSampler(valset, shuffle=False) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=getattr(hparams, 'num_workers', 1), # <<< CHANGE: Use hparam >>>
                                shuffle=False, batch_size=batch_size,
                                pin_memory=getattr(hparams, 'pin_memory', False), # <<< CHANGE: Use hparam >>>
                                collate_fn=collate_fn, drop_last=False) # <<< CHANGE: Typically don't drop last in validation >>>

        val_loss = 0.0
        num_batches = len(val_loader)
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch) # Assume parse_batch handles moving data to GPU
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                # Sum losses across GPUs, then average by world size
                dist.all_reduce(loss, op=dist.ReduceOp.SUM) # <<< CHANGE: Use newer enum if available >>>
                reduced_val_loss = (loss / n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss # Accumulate loss per batch

        # Average loss over all validation batches
        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0


    model.train() # Set model back to training mode


    # <<< CHANGE: Return the calculated average validation loss >>>
    # Logging is now handled in the main train loop
    # if rank == 0:
    #     print("Validation loss {}: {:9f}  ".format(iteration, avg_val_loss))
    #     # logger.log_validation(avg_val_loss, model, y, y_pred, iteration) # logger might need update for new signature

    return avg_val_loss


# <<< CHANGE: Updated train function signature and added logic >>>
def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams, use_wandb=False): # <<< CHANGE: Added use_wandb flag >>>
    """Training and validation logging results to tensorboard and stdout/wandb"""

    if getattr(hparams, 'distributed_run', False): # <<< CHANGE: Safer access >>>
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay,
                                 betas=getattr(hparams, 'betas', (0.9, 0.999)), # <<< CHANGE: Use hparam if available >>>
                                 eps=getattr(hparams, 'eps', 1e-08)) # <<< CHANGE: Use hparam if available >>>


    # --- FP16/AMP Setup ---
    scaler = None
    if getattr(hparams, 'fp16_run', False): # <<< CHANGE: Safer access >>>
        try:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            print("Using PyTorch AMP (fp16) with GradScaler.")
        except ImportError:
             # Fallback or error if torch.cuda.amp not available
             print("Warning: torch.cuda.amp not found for fp16_run. FP16 disabled.")
             hparams.fp16_run = False


    # --- Distributed Data Parallel (DDP) Setup ---
    # Recommended over apply_gradient_allreduce for efficiency
    if getattr(hparams, 'distributed_run', False):
        try:
           
            # find_unused_parameters=True can be necessary for some models but adds overhead
            model = DDP(model, device_ids=[rank % torch.cuda.device_count()], find_unused_parameters=False)
            print(f"Wrapped model with DistributedDataParallel on rank {rank}.")
        except ImportError:
            print("Warning: Could not wrap model with DDP. Using apply_gradient_allreduce (less efficient).")
            # Fallback to original method if DDP fails or isn't desired
            model = apply_gradient_allreduce(model)


    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank) # <<< CHANGE: Pass hparams >>>

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    # <<< CHANGE: Early stopping state initialization >>>
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_checkpoint_path = None
    # Retrieve early stopping params from hparams, provide defaults
    early_stopping_patience = getattr(hparams, 'early_stopping_patience', 10)
    min_val_loss_delta = getattr(hparams, 'min_val_loss_delta', 0.0001)


    if checkpoint_path is not None and os.path.isfile(checkpoint_path): # <<< CHANGE: Check if file exists >>>
        if warm_start:
            # Note: Warm start usually ignores optimizer state and iteration count
            model_to_load = model.module if isinstance(model, DDP) else model # Load into the underlying module if DDP wrapped
            model_to_load = warm_start_model(checkpoint_path, model_to_load, getattr(hparams, 'ignore_layers', []))
            print("Warm start complete. Optimizer and iteration count reset.")
            # Reset learning rate explicitly if needed for warm start
            for param_group in optimizer.param_groups:
                 param_group['lr'] = learning_rate
        else:
            # Resuming training
            model_to_load = model.module if isinstance(model, DDP) else model
            model_to_load, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model_to_load, optimizer)
            # Load the checkpoint into the main model variable (handles DDP wrapper state)
            if isinstance(model, DDP): model.module.load_state_dict(model_to_load.state_dict())
            else: model.load_state_dict(model_to_load.state_dict())

            if hparams.use_saved_learning_rate and _learning_rate is not None:
                learning_rate = _learning_rate
            # Apply the loaded learning rate to the optimizer
            for param_group in optimizer.param_groups:
                 param_group['lr'] = learning_rate

            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))
            print(f"Resuming from iteration {iteration}, epoch offset {epoch_offset}")

            # <<< CHANGE: Optionally load best_val_loss from checkpoint if saved >>>
            # checkpoint_dict_for_resume = torch.load(checkpoint_path, map_location='cpu')
            # best_val_loss = checkpoint_dict_for_resume.get('best_val_loss', float('inf'))
            # print(f"Resuming with best_val_loss: {best_val_loss}")

    else:
         print("No valid checkpoint path provided or file not found. Starting from scratch.")


    model.train()

    # ================ MAIN TRAINING LOOP! ===================
    train_start_time = time.time()
    print(f"Starting training for {hparams.epochs - epoch_offset} epochs...")

    for epoch in range(epoch_offset, hparams.epochs):
        epoch_start_time = time.time()
        print(f"--- Epoch: {epoch} ---")

        # Set epoch for distributed sampler (important!)
        if hparams.distributed_run and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # --- Training Phase ---
        model.train() # Ensure model is in train mode
        accumulated_loss = 0.0
        accumulated_grad_norm = 0.0
        batch_count_in_epoch = 0

        for i, batch in enumerate(train_loader):
            start = time.perf_counter()

            # Adjust learning rate? (Example: learning rate scheduling)
            # learning_rate = adjust_learning_rate(optimizer, iteration, hparams)
            # --- Set current learning rate in optimizer ---
            # Should be done only if LR scheduling is used, otherwise LR is fixed or loaded
            # for param_group in optimizer.param_groups:
            #      param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch) # Assumes this moves data to cuda:rank

            # --- Forward pass with AMP context if enabled ---
            if getattr(hparams, 'fp16_run', False) and scaler is not None:
                with torch.cuda.amp.autocast():
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
            else: # Normal FP32 forward pass
                y_pred = model(x)
                loss = criterion(y_pred, y)

            # --- Loss reduction for distributed training ---
            if hparams.distributed_run:
                 # Average the loss across GPUs
                 dist.all_reduce(loss, op=dist.ReduceOp.AVG) # <<< CHANGE: Use AVG for loss >>>
            reduced_loss = loss.item() # Get scalar value after reduction
            accumulated_loss += reduced_loss
            batch_count_in_epoch += 1

            # --- Backward pass ---
            if getattr(hparams, 'fp16_run', False) and scaler is not None:
                scaler.scale(loss).backward() # Scales loss, calls backward
            else:
                loss.backward() # Standard backward pass

            # --- Gradient Clipping and Optimizer Step ---
            grad_norm = 0.0
            if getattr(hparams, 'fp16_run', False) and scaler is not None:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                # Clip gradients (applied to model.parameters() directly or amp master params if using apex)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), getattr(hparams, 'grad_clip_thresh', 1.0)) # <<< CHANGE: Use hparam >>>
                # Optimizer step - scaler checks for overflows/underflows
                scaler.step(optimizer)
                # Update scaler for next iteration
                scaler.update()
            else: # Normal FP32 gradient clipping and step
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), getattr(hparams, 'grad_clip_thresh', 1.0)) # <<< CHANGE: Use hparam >>>
                optimizer.step()

            # Check for NaN grad_norm (overflow detection)
            is_overflow = math.isnan(grad_norm) or math.isinf(grad_norm)
            if not is_overflow:
                 accumulated_grad_norm += grad_norm # Accumulate valid grad norms

            # --- Logging (only on rank 0) ---
            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                # Log less frequently than every step to avoid excessive output/wandb calls
                log_interval = getattr(hparams, 'log_interval', 100) # <<< CHANGE: Add hparam for log interval >>>
                if iteration % log_interval == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print("Train Iter: {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tGrad Norm: {:.6f}\tLR: {:.1E}\tTime: {:.2f}s/it".format(
                        iteration, epoch, i * hparams.batch_size * n_gpus, len(train_loader.dataset), # Approximate progress
                        100. * i / len(train_loader), reduced_loss, grad_norm, current_lr, duration))

                    # --- W&B Logging (Training) ---
                    if use_wandb:
                        wandb.log({
                            'train/loss': reduced_loss,
                            'train/grad_norm': grad_norm,
                            'train/learning_rate': current_lr,
                            'train/epoch': epoch + (i / len(train_loader)), # Log fractional epoch
                            # 'train/duration_per_iter': duration
                        }, step=iteration)

                    # --- TensorBoard Logging ---
                    if logger is not None:
                        logger.log_training(reduced_loss, grad_norm, current_lr, duration, iteration)

            # <<< CHANGE: Moved Validation and Checkpointing outside inner loop, run per epoch or N iterations >>>

            iteration += 1 # Increment iteration counter

            # --- End of Batch Loop ---

        # --- End of Epoch Calculations & Logging (Rank 0) ---
        if rank == 0:
             avg_epoch_loss = accumulated_loss / batch_count_in_epoch if batch_count_in_epoch > 0 else 0.0
             avg_epoch_grad_norm = accumulated_grad_norm / batch_count_in_epoch if batch_count_in_epoch > 0 else 0.0
             epoch_duration = time.time() - epoch_start_time
             print(f"--- Epoch {epoch} Summary ---")
             print(f"Average Training Loss: {avg_epoch_loss:.6f}")
             print(f"Average Grad Norm: {avg_epoch_grad_norm:.6f}")
             print(f"Epoch Duration: {epoch_duration:.2f}s")
             if use_wandb:
                  wandb.log({
                       'epoch/train_loss': avg_epoch_loss,
                       'epoch/grad_norm': avg_epoch_grad_norm,
                       'epoch/duration': epoch_duration,
                       'epoch': epoch # Log whole epoch number
                  }, step=iteration) # Log summary at the end of the epoch's iterations
             # --- TensorBoard Logging (Epoch Summary) ---
             if logger is not None:
                 logger.add_scalar("training.epoch_loss_by_epoch", avg_epoch_loss, epoch)
                 logger.add_scalar("grad.epoch_norm_by_epoch", avg_epoch_grad_norm, epoch)


        # --- Validation, Checkpointing, and Early Stopping (Periodically, e.g., end of epoch) ---
        # Run validation every `validate_every_n_epochs` or `iters_per_checkpoint`
        # iters_per_checkpoint = getattr(hparams, 'iters_per_checkpoint', 1000) # <<< CHANGE: Use hparam >>>
        # run_validation_checkpoint = (iteration % iters_per_checkpoint == 0) # Or check epoch interval

        # Perform validation only on rank 0 after sync if using DDP
        if rank == 0:
             print(f"\n--- Running Validation at Iteration {iteration} ---")
             val_start_time = time.time()
             # Use the underlying model for validation if wrapped
             model_to_validate = model.module if isinstance(model, DDP) else model
             # <<< CHANGE: Call validate and get loss >>>
             val_loss = validate(model_to_validate, criterion, valset, iteration,
                                 hparams.batch_size, n_gpus, collate_fn, # Pass batch_size for val loader
                                 hparams.distributed_run, rank, hparams) # Pass hparams
             val_duration = time.time() - val_start_time
             print(f"Validation Loss: {val_loss:.6f} (Duration: {val_duration:.2f}s)")

             # --- W&B Logging (Validation) ---
             if use_wandb:
                 wandb.log({
                     'val/loss': val_loss, # <<< Log the returned loss >>>
                     'epoch': epoch # Can log epoch here too
                 }, step=iteration)

             # --- TensorBoard Logging (Validation) ---
             # if logger is not None:
             #     # Assuming log_validation takes loss, model, maybe examples, iteration
             #     # You might need to fetch a validation batch again to log examples
             #     # For simplicity, just logging the loss value here:
                 
             #     logger.log_validation(val_loss, iteration=iteration) # <<< Adjust logger call based on its definition >>>


             # --- Regular Checkpointing ---
             checkpoint_path_iter = os.path.join(
                 output_directory, "checkpoint_{}".format(iteration))
             save_checkpoint(model, optimizer, learning_rate, iteration,
                             checkpoint_path_iter, hparams=hparams, best_val_loss=best_val_loss) # <<< Pass hparams and best_val_loss >>>


             # --- Early Stopping Logic ---
             print(f"Checking Early Stopping: Current Best Loss = {best_val_loss:.6f}, New Loss = {val_loss:.6f}")
             if val_loss < best_val_loss - min_val_loss_delta:
                 best_val_loss = val_loss
                 patience_counter = 0 # Reset patience
                 # Save this as the 'best' model checkpoint
                 best_model_checkpoint_path_new = os.path.join(output_directory, f"checkpoint_best_iter_{iteration}_loss_{val_loss:.4f}.pt") #<<< CHANGE: Added .pt extension >>>
                 save_checkpoint(model, optimizer, learning_rate, iteration, best_model_checkpoint_path_new, hparams=hparams, best_val_loss=best_val_loss)
                 print(f"  Validation loss improved! New best: {best_val_loss:.6f}. Saved best model: {best_model_checkpoint_path_new}")
                 # Keep track of the latest best path
                 best_model_checkpoint_path = best_model_checkpoint_path_new
                 # Clean up older "best" checkpoints if desired
             else:
                 patience_counter += 1
                 print(f"  Validation loss did not improve significantly ({patience_counter}/{early_stopping_patience}).")

             if patience_counter >= early_stopping_patience:
                 print(f"--- EARLY STOPPING TRIGGERED after {patience_counter} checks without improvement. ---")
                 print(f"Epoch: {epoch}, Iteration: {iteration}. Best Validation Loss: {best_val_loss:.6f}")
                 if best_model_checkpoint_path:
                      print(f"Best model saved at: {best_model_checkpoint_path}")
                 else:
                      print("No best model was saved (validation may not have improved).")
                 break # Exit the outer epoch loop

             print("--- Validation Complete ---")
             # --- End of Rank 0 Validation/Checkpointing Block ---

        # Ensure all processes sync before starting next epoch if using DDP
        if hparams.distributed_run:
             dist.barrier()

        # --- Check if early stopping break occurred (all ranks need to break) ---
        stop_training = torch.tensor(0).cuda() # Default is don't stop
        if rank == 0 and patience_counter >= early_stopping_patience:
             stop_training = torch.tensor(1).cuda() # Signal to stop
        if hparams.distributed_run:
             dist.broadcast(stop_training, src=0) # Broadcast stop signal from rank 0
        if stop_training.item() == 1:
             print(f"Rank {rank} received early stopping signal. Breaking epoch loop.")
             break # All ranks break the epoch loop

    # --- End of Training Loop ---
    if rank == 0:
        print("\n=============== Training Finished ===============")
        total_training_time = time.time() - train_start_time
        print(f"Total Training Time: {total_training_time:.2f}s")
        if best_model_checkpoint_path:
             print(f"Final best model checkpoint saved at: {best_model_checkpoint_path}")
        else:
             print("No 'best' checkpoint was saved during training (validation may not have improved).")
        # Optionally save the final model state regardless of early stopping
        final_checkpoint_path = os.path.join(output_directory, f"checkpoint_final_{iteration}.pt") # <<< CHANGE: Added .pt extension >>>
        save_checkpoint(model, optimizer, learning_rate, iteration, final_checkpoint_path, hparams=hparams, best_val_loss=best_val_loss)
        print(f"Final model state saved at: {final_checkpoint_path}")
        print("=================================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, required=True, # <<< CHANGE: Made required >>>
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str, required=True, # <<< CHANGE: Made required >>>
                        help='directory to save tensorboard logs (relative to output_directory)')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path for resuming or warm-starting')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers and optimizer/iteration')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str, default="", # <<< CHANGE: Default to empty string >>>
                        required=False, help='comma separated name=value pairs')
    # <<< CHANGE: Add use_wandb argument controlled by the calling script >>>
    # This script itself shouldn't parse use_wandb, it should be passed by the calling function
    # parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')


    args = parser.parse_args()
    # <<< CHANGE: Pass hparams string directly to create_hparams >>>
    hparams = create_hparams(args.hparams)

    # --- Set CUDNN flags ---
    torch.backends.cudnn.enabled = getattr(hparams, 'cudnn_enabled', True) # <<< CHANGE: Safer access w/ default >>>
    torch.backends.cudnn.benchmark = getattr(hparams, 'cudnn_benchmark', False) # <<< CHANGE: Safer access w/ default >>>

    print("--- Hyperparameters ---")
    for k, v in vars(hparams).items(): # <<< CHANGE: Iterate hparams directly if it's an object/dict >>>
        print(f"  {k}: {v}")
    print("-----------------------")

    print("FP16 Run:", getattr(hparams, 'fp16_run', False))
    print("Dynamic Loss Scaling:", getattr(hparams, 'dynamic_loss_scaling', True)) # Often handled by AMP scaler
    print("Distributed Run:", getattr(hparams, 'distributed_run', False))
    print("cuDNN Enabled:", torch.backends.cudnn.enabled)
    print("cuDNN Benchmark:", torch.backends.cudnn.benchmark)

    # <<< CHANGE: The 'use_wandb' flag is now passed programmatically from your sweep script >>>
    # train(args.output_directory, args.log_directory, args.checkpoint_path,
    #       args.warm_start, args.n_gpus, args.rank, args.group_name, hparams, args.use_wandb)

    # <<< This __main__ block is usually run when executing train.py directly >>>
    # <<< For W&B sweeps, the agent calls the run_training_sweep function in your *other* script >>>
    # <<< which then calls the train() function defined here. >>>
    # <<< You might keep this for direct testing, passing a default use_wandb=False >>>
    train(args.output_directory, args.log_directory, args.checkpoint_path,
           args.warm_start, args.n_gpus, args.rank, args.group_name, hparams, use_wandb=False) # Default to False if run directly
