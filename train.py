import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import math
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim.lr_scheduler import OneCycleLR
torch.autograd.set_detect_anomaly(True)
from dataset import get_dataset, get_tokenizer
from transcribe_model import TranscribeModel
from torch import nn

vq_initial_loss_weight = 10
vq_warmup_steps = 1000
vq_final_loss_weight = 0.5
num_epochs = 1000
starting_steps = 0
num_examples = None
model_id = "test21"
num_batch_repeats = 1

starting_steps = 0
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

def run_loss_function(log_probs, target, blank_token):
    loss_function = nn.CTCLoss(blank=blank_token)
    
    input_lengths = torch.full((log_probs.shape[0],), log_probs.shape[1], 
                              dtype=torch.long, device=log_probs.device)
    
    # Use torch.ne for element-wise comparison
    target_lengths = torch.ne(target, blank_token).sum(dim=1).to(torch.long)
    
    input_seq_first = log_probs.permute(1, 0, 2)
    loss = loss_function(input_seq_first, target, input_lengths, target_lengths)
    return loss

def safe_mean(losses):
    """Calculate mean safely, handling empty lists"""
    return sum(losses) / len(losses) if len(losses) > 0 else 0.0

# def main():
#     log_dir = f"runs/speech2text_training/{model_id}"
#     if os.path.exists(log_dir):
#         import shutil
        
#         shutil.rmtree(log_dir)
#     writer = SummaryWriter(log_dir)
    
#     tokenizer = get_tokenizer()
#     blank_token = tokenizer.token_to_id("<□>")
    
#     device = torch.device(
#         "cuda" 
#         if torch.cuda.is_available() 
#         else "mps" if torch.backends.mps.is_available() else "cpu"
#     )
#     print(f"Using device: {device}")
    
    

    
#     if os.path.exists(r"C:\Users\Kamil\Desktop\Coding\models\test21\model_step_3000.pth"):
#         print(f"Loading model from models/{model_id}/model_latest.pth")
        
#         model = TranscribeModel.load(r"C:\Users\Kamil\Desktop\Coding\models\test21\model_step_3000.pth").to(device)
#     else:
#         model = TranscribeModel( 
#             num_codebooks=4,  # Increased from 2
#             codebook_size=64,  # Increased from 32
#             embedding_dim=256,  # Increased from 16 - this was too small
#             num_transformer_layers=6,  # Increased from 2
#             vocab_size=len(tokenizer.get_vocab()),
#             strides=[4, 4, 4, 2],  # Better stride configuration
#             initial_mean_pooling_kernel_size=2,  # Reduced from 4
#             max_seq_length=800,
#         ).to(device)
        
#     num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Number of trainable parameters: {num_trainable_params}")
    
#     optimizer = torch.optim.AdamW( 
#     model.parameters(), 
#     lr=LEARNING_RATE,
#     weight_decay=0.01,  
#     betas=(0.9, 0.98)   
# )
  
#     dataloader = get_dataset(
#         batch_size=BATCH_SIZE,
#         num_examples=num_examples,
#         num_workers=0,
#     )
#     scheduler = OneCycleLR(
#         optimizer,
#         max_lr=LEARNING_RATE,
#         steps_per_epoch=len(dataloader),
#         epochs=num_epochs,
#         pct_start=0.1  # Warm up for 10% of training
#     )  
#     ctc_losses = []
#     vq_losses = []
#     num_batches = len(dataloader)
#     steps = starting_steps

#     # Create directory for saving models if it doesn't exist
#     os.makedirs(f"models/{model_id}", exist_ok=True)

#     vq_initial_loss_weight = 1.0  # Reduced from 10
#     vq_warmup_steps = 2000  # Increased warmup
#     vq_final_loss_weight = 0.1  # Reduced final weight

# # Add gradient accumulation for larger effective batch size
#     gradient_accumulation_steps = 4
#     effective_batch_size = BATCH_SIZE * gradient_accumulation_steps

# # Initialize loss tracking
#     ctc_losses = []
#     vq_losses = []
#     steps = starting_steps

import jiwer
from tqdm import tqdm



def greedy_decoder(log_probs, blank_token=0):
    """Improved greedy decoder for CTC outputs."""
    # Get the most likely token at each timestep
    predictions = torch.argmax(log_probs, dim=-1).cpu().numpy()
    decoded_predictions = []
    
    for pred in predictions:
        # Remove consecutive duplicates and blanks
        previous = -1
        decoded_seq = []
        for p in pred:
            if p != previous and p != blank_token:
                decoded_seq.append(p)
            previous = p
        decoded_predictions.append(decoded_seq)
    
    return decoded_predictions


def calculate_wer(predictions, references):
    """Calculate Word Error Rate between predictions and references."""
    try:
        return jiwer.wer(references, predictions)
    except:
        return 1.0

def calculate_cer(predictions, references):
    """Calculate Character Error Rate between predictions and references."""
    total_chars = sum(len(ref) for ref in references)
    total_edits = 0
    
    for pred, ref in zip(predictions, references):
        # Simple Levenshtein distance calculation
        dp = [[0] * (len(ref) + 1) for _ in range(len(pred) + 1)]
        
        for i in range(len(pred) + 1):
            dp[i][0] = i
        for j in range(len(ref) + 1):
            dp[0][j] = j
            
        for i in range(1, len(pred) + 1):
            for j in range(1, len(ref) + 1):
                if pred[i-1] == ref[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        total_edits += dp[len(pred)][len(ref)]
    
    return total_edits / total_chars if total_chars > 0 else 1.0

def evaluate_model(model, dataloader, tokenizer, device, blank_token, max_batches=5):
    """Evaluate the model and return metrics with sample predictions."""
    model.eval()
    all_predictions = []
    all_references = []
    sample_examples = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
                
            audio = batch["audio"].to(device)
            target = batch["input_ids"].to(device)
            text = batch["text"]
            
            # Forward pass
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
            
            output, _ = model(audio)
            blank_token_id = blank_token
            # Decode predictions - use the correct blank token
            decoded_preds = greedy_decoder(output, blank_token=blank_token)
            
            # Convert token IDs to text - FIXED VERSION
            pred_texts = []
            for pred in decoded_preds:
                tokens = []
                for p in pred:
                    if p < len(tokenizer.get_vocab()) and p != blank_token_id:
                        token = tokenizer.id_to_token(p)
                        # Filter out ALL special tokens
                        if token and token not in ["<pad>", "<unk>", "<s>", "</s>", "<□>"]:
                            tokens.append(token)
                pred_text = "".join(tokens)
                pred_texts.append(pred_text)
                
            all_predictions.extend(pred_texts)     
            all_references.extend(text) 
            # Store first few examples for display
            if i < 3:
                for j, (pred, ref) in enumerate(zip(pred_texts, text)):
                    if len(sample_examples) < 6:
                        sample_examples.append({
                            'reference': ref,
                            'prediction': pred,
                            'batch': i,
                            'sample': j
                        })
    
    model.train()
    
    # Calculate metrics
    wer = calculate_wer(all_predictions, all_references)
    cer = calculate_cer(all_predictions, all_references)
    
    return {
        'wer': wer,
        'cer': cer,
        'num_samples': len(all_predictions),
        'examples': sample_examples
    }


def print_evaluation_results(eval_results, step):
    """Print evaluation results in a nice format."""
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS AT STEP {step}")
    print("="*80)
    print(f"Word Error Rate (WER): {eval_results['wer']:.4f}")
    print(f"Character Error Rate (CER): {eval_results['cer']:.4f}")
    print(f"Number of samples evaluated: {eval_results['num_samples']}")
    print("\nSAMPLE PREDICTIONS:")
    print("-"*80)
    
    for i, example in enumerate(eval_results['examples']):
        print(f"\nExample {i+1}:")
        print(f"Reference:  '{example['reference']}'")
        print(f"Prediction: '{example['prediction']}'")
        
        # Calculate individual WER for this example
        individual_wer = calculate_wer([example['prediction']], [example['reference']])
        print(f"Individual WER: {individual_wer:.4f}")
    
    print("="*80 + "\n")

def main():
    log_dir = f"runs/speech2text_training/{model_id}"
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    tokenizer = get_tokenizer()
    blank_token = tokenizer.token_to_id("<□>")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load or create model
    if os.path.exists(r"C:\Users\Kamil\Desktop\Coding\models\test21\model_step_3000.pth"):
        print(f"Loading model from models/{model_id}/model_step_3000.pth")
        model = TranscribeModel.load(r"C:\Users\Kamil\Desktop\Coding\models\test21\model_step_3000.pth").to(device)
    else:
        model = TranscribeModel(
            num_codebooks=2,        # Reduced from 4
            codebook_size=32,       # Reduced from 64
            embedding_dim=128,      # Reduced from 256
            num_transformer_layers=3, # Reduced from 6
            vocab_size=len(tokenizer.get_vocab()),
            strides=[4, 4, 2],      # Simplified from [4,4,4,2]
            initial_mean_pooling_kernel_size=2,
            max_seq_length=800,
        ).to(device)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )

    dataloader = get_dataset(
        batch_size=BATCH_SIZE,
        num_examples=num_examples,
        num_workers=0,
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(dataloader),
        epochs=num_epochs,
        pct_start=0.1
    )

    # Create evaluation dataloader (smaller batch size for evaluation)
    eval_dataloader = get_dataset(
        batch_size=16,
        num_examples=100,  # Evaluate on 100 samples
        num_workers=0,
    )

    # Training configuration
    vq_initial_loss_weight = 0.1
    vq_warmup_steps = 2000
    vq_final_loss_weight = 0.01
    gradient_accumulation_steps = 4
    
    # Initialize loss tracking
    ctc_losses = []
    vq_losses = []
    steps = starting_steps
    
    # Create directory for saving models
    os.makedirs(f"models/{model_id}", exist_ok=True)

    print("Starting training...")
    print(f"Total steps per epoch: {len(dataloader)}")
    print(f"Evaluation every 1000 steps")
    
    for i in range(num_epochs):
        model.train()
        epoch_start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        epoch_end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if epoch_start_time:
            epoch_start_time.record()
        
        for idx, batch in enumerate(dataloader):
            audio = batch["audio"].to(device)
            target = batch["input_ids"].to(device)

            if audio.dim() == 2:
                audio = audio.unsqueeze(1)

            output, vq_loss = model(audio)
            ctc_loss = run_loss_function(output, target, blank_token)

            # Improved loss weighting with cosine annealing
            progress = steps / (num_epochs * len(dataloader))
            vq_weight = vq_final_loss_weight + (vq_initial_loss_weight - vq_final_loss_weight) * \
                       (1 + math.cos(math.pi * min(progress * 2, 1))) / 2

            if vq_loss is not None:
                total_loss = ctc_loss + vq_weight * vq_loss
            else:
                total_loss = ctc_loss

            total_loss = total_loss / gradient_accumulation_steps
            total_loss.backward()

            ctc_losses.append(ctc_loss.item())
            vq_losses.append(vq_loss.item() if vq_loss is not None else 0.0)

            if (idx + 1) % gradient_accumulation_steps == 0 or (idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            steps += 1

            # Regular logging every 20 steps
            if steps % 20 == 0:
                if len(ctc_losses) > 0:
                    avg_ctc_loss = safe_mean(ctc_losses)
                    avg_vq_loss = safe_mean(vq_losses)
                    avg_loss = avg_ctc_loss + vq_weight * avg_vq_loss

                    print(
                        f"Epoch {i}, Batch {idx}/{len(dataloader)}, Step {steps}, "
                        f"Loss: {avg_loss:.4f}, CTC Loss: {avg_ctc_loss:.4f}, "
                        f"VQ Loss: {avg_vq_loss:.4f}, VQ Weight: {vq_weight:.4f}"
                    )

                    writer.add_scalar("Loss/train", avg_loss, steps)
                    writer.add_scalar("Loss/ctc", avg_ctc_loss, steps)
                    writer.add_scalar("Loss/vq", avg_vq_loss, steps)
                    writer.add_scalar("Loss/vq_weight", vq_weight, steps)

                    ctc_losses = []
                    vq_losses = []

            # Evaluation every 1000 steps
            if steps % 1000 == 0:
                print(f"\nRunning evaluation at step {steps}...")
                eval_results = evaluate_model(model, eval_dataloader, tokenizer, device, blank_token)
                
                # Log metrics to tensorboard
                writer.add_scalar("Metrics/WER", eval_results['wer'], steps)
                writer.add_scalar("Metrics/CER", eval_results['cer'], steps)
                
                # Print detailed results
                print_evaluation_results(eval_results, steps)

            # Save model periodically
            if steps % 500 == 0:
                model_path = f"models/{model_id}/model_step_{steps}.pth"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)

                try:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': steps,
                        'epoch': i,
                        'vq_weight': vq_weight,
                    }, model_path)
                    print(f"Model saved to {model_path}")

                    latest_path = f"models/{model_id}/model_latest.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': steps,
                        'epoch': i,
                        'vq_weight': vq_weight,
                    }, latest_path)

                except Exception as e:
                    print(f"Error saving model: {e}")

        if epoch_end_time:
            epoch_end_time.record()
            torch.cuda.synchronize()
            epoch_time = epoch_start_time.elapsed_time(epoch_end_time) / 1000.0
            print(f"Epoch {i} completed in {epoch_time:.2f} seconds")

    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    final_eval_results = evaluate_model(model, eval_dataloader, tokenizer, device, blank_token, max_batches=10)
    print_evaluation_results(final_eval_results, steps)

    # Save final model
    try:
        final_path = f"models/{model_id}/model_final.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': steps,
            'epoch': num_epochs,
            'final_wer': final_eval_results['wer'],
            'final_cer': final_eval_results['cer'],
        }, final_path)
        print(f"Final model saved to {final_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")

    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()