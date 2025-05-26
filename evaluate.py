import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataset, get_tokenizer
from transcribe_model import TranscribeModel
from jiwer import wer
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import sounddevice as sd
import librosa
import argparse

def calculate_wer(predictions, references):
    """Calculate Word Error Rate between predictions and references."""
    return wer(references, predictions)

def calculate_cer(predictions, references):
    """Calculate Character Error Rate between predictions and references."""
    total_chars = sum(len(ref) for ref in references)
    total_edits = 0
    
    for pred, ref in zip(predictions, references):
        # Calculate Levenshtein distance
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

def greedy_decoder(log_probs, blank_token=0):
    """
    Greedy decoder for CTC outputs.
    
    Args:
        log_probs: Log probabilities from model output, shape [batch, time, vocab]
        blank_token: Index of the blank token
        
    Returns:
        List of lists containing token indices for each sequence in the batch
    """
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

def evaluate_model(model, dataloader, tokenizer, device, max_batches=None):
    """
    Evaluate the model on the given dataloader.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with evaluation data
        tokenizer: Tokenizer for decoding predictions
        device: Device to run evaluation on
        max_batches: Maximum number of batches to evaluate (None for all)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_references = []
    all_wers = []
    all_cers = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches is not None and i >= max_batches:
                break
                
            audio = batch["audio"].to(device)
            target = batch["input_ids"].to(device)
            text = batch["text"]
            
            # Forward pass
            output, _ = model(audio)
            
            # Decode predictions
            blank_token_id = tokenizer.token_to_id("<□>") if "<□>" in tokenizer.get_vocab() else 0
            decoded_preds = greedy_decoder(output, blank_token=blank_token_id)
            
            # Convert token IDs to text
            pred_texts = []
            for pred in decoded_preds:
                tokens = [tokenizer.id_to_token(p) for p in pred if p < len(tokenizer.get_vocab())]
                pred_texts.append("".join(tokens))
            
            # Calculate per-sample metrics
            for pred, ref in zip(pred_texts, text):
                all_wers.append(calculate_wer([pred], [ref]))
                all_cers.append(calculate_cer([pred], [ref]))
            
            all_predictions.extend(pred_texts)
            all_references.extend(text)
    
    # Calculate overall metrics
    word_error_rate = calculate_wer(all_predictions, all_references)
    char_error_rate = calculate_cer(all_predictions, all_references)
    
    return {
        "wer": word_error_rate,
        "cer": char_error_rate,
        "per_sample_wer": all_wers,
        "per_sample_cer": all_cers,
        "num_samples": len(all_predictions),
        "predictions": all_predictions,
        "references": all_references
    }

def plot_error_distribution(results, output_dir="evaluation_results"):
    """Plot distribution of WER and CER across samples."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(results["per_sample_wer"], kde=True)
    plt.title(f"Word Error Rate Distribution\nMean: {np.mean(results['per_sample_wer']):.4f}")
    plt.xlabel("WER")
    plt.ylabel("Count")
    
    plt.subplot(1, 2, 2)
    sns.histplot(results["per_sample_cer"], kde=True)
    plt.title(f"Character Error Rate Distribution\nMean: {np.mean(results['per_sample_cer']):.4f}")
    plt.xlabel("CER")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_distributions.png"))
    plt.close()

def plot_confusion_matrix(results, top_n=20, output_dir="evaluation_results"):
    """Plot confusion matrix for most common errors."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect character-level errors
    char_errors = {}
    for pred, ref in zip(results["predictions"], results["references"]):
        # Simple alignment using minimum edit distance
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
        
        # Backtrack to find alignments and errors
        i, j = len(pred), len(ref)
        while i > 0 or j > 0:
            if i > 0 and j > 0 and pred[i-1] == ref[j-1]:
                i -= 1
                j -= 1
            else:
                min_val = float('inf')
                if i > 0:
                    min_val = min(min_val, dp[i-1][j])
                if j > 0:
                    min_val = min(min_val, dp[i][j-1])
                if i > 0 and j > 0:
                    min_val = min(min_val, dp[i-1][j-1])
                
                if i > 0 and j > 0 and dp[i-1][j-1] == min_val:
                    # Substitution
                    error_key = f"{ref[j-1]}->{pred[i-1]}"
                    char_errors[error_key] = char_errors.get(error_key, 0) + 1
                    i -= 1
                    j -= 1
                elif i > 0 and dp[i-1][j] == min_val:
                    # Insertion
                    error_key = f"->'{pred[i-1]}'"
                    char_errors[error_key] = char_errors.get(error_key, 0) + 1
                    i -= 1
                elif j > 0 and dp[i][j-1] == min_val:
                    # Deletion
                    error_key = f"'{ref[j-1]}'->"
                    char_errors[error_key] = char_errors.get(error_key, 0) + 1
                    j -= 1
    
    # Plot top N errors
    top_errors = sorted(char_errors.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    plt.figure(figsize=(12, 8))
    error_types = [err[0] for err in top_errors]
    error_counts = [err[1] for err in top_errors]
    
    plt.barh(error_types, error_counts)
    plt.xlabel("Count")
    plt.ylabel("Error Type")
    plt.title(f"Top {top_n} Most Common Errors")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "common_errors.png"))
    plt.close()

def save_examples(results, num_examples=10, output_dir="evaluation_results"):
    """Save examples of predictions vs references."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate error for each example
    examples_with_errors = []
    for i, (pred, ref) in enumerate(zip(results["predictions"], results["references"])):
        wer_val = results["per_sample_wer"][i]
        cer_val = results["per_sample_cer"][i]
        examples_with_errors.append((i, pred, ref, wer_val, cer_val))
    
    # Sort by WER
    examples_with_errors.sort(key=lambda x: x[3])
    
    # Get best and worst examples
    best_examples = examples_with_errors[:num_examples//2]
    worst_examples = examples_with_errors[-num_examples//2:]
    selected_examples = best_examples + worst_examples
    
    # Save to file
    with open(os.path.join(output_dir, "example_predictions.txt"), "w", encoding="utf-8") as f:
        f.write("BEST EXAMPLES:\n")
        f.write("=" * 80 + "\n\n")
        for i, pred, ref, wer_val, cer_val in best_examples:
            f.write(f"Example {i}:\n")
            f.write(f"Reference: {ref}\n")
            f.write(f"Prediction: {pred}\n")
            f.write(f"WER: {wer_val:.4f}, CER: {cer_val:.4f}\n\n")
        
        f.write("\nWORST EXAMPLES:\n")
        f.write("=" * 80 + "\n\n")
        for i, pred, ref, wer_val, cer_val in worst_examples:
            f.write(f"Example {i}:\n")
            f.write(f"Reference: {ref}\n")
            f.write(f"Prediction: {pred}\n")
            f.write(f"WER: {wer_val:.4f}, CER: {cer_val:.4f}\n\n")
    
    return selected_examples

def transcribe_audio_file(model, tokenizer, audio_file, device):
    """Transcribe a single audio file."""
    # Load audio file
    audio, sr = librosa.load(audio_file, sr=16000)
    
    # Convert to tensor
    audio_tensor = torch.tensor(audio).float().unsqueeze(0).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output, _ = model(audio_tensor)
    
    # Decode prediction
    blank_token_id = tokenizer.token_to_id("□") if "□" in tokenizer.get_vocab() else 0
    decoded_pred = greedy_decoder(output, blank_token=blank_token_id)[0]
    
    # Convert token IDs to text
    tokens = [tokenizer.id_to_token(p) for p in decoded_pred if p < len(tokenizer.get_vocab())]
    transcription = "".join(tokens)
    
    return transcription

def main():
    parser = argparse.ArgumentParser(description="Evaluate speech recognition model")
    parser.add_argument("--model_id", type=str, default="test21", help="Model ID to evaluate")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--audio_file", type=str, help="Path to audio file for transcription")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.backends.mps.is_available() 
                         else "cpu")
    print(f"Using device: {device}")
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Determine model path
    model_path = args.model_path
    if model_path is None:
        model_path =r"C:\Users\Kamil\Desktop\Coding\models\test21\model_step_20500.pth"
        if not os.path.exists(model_path):
            model_path = r"C:\Users\Kamil\Desktop\Coding\models\test21\model_step_20500.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}")
    
    # Create model instance first
    model = TranscribeModel(
        num_codebooks=4,        # Zwiększ z 2
        codebook_size=64,       # Zwiększ z 32
        embedding_dim=256,      # Zwiększ z 128
        num_transformer_layers=6, # Zwiększ z 3
        vocab_size=len(tokenizer.get_vocab()),  # DODANE - wymagane
        strides=[8, 8, 4],      # Bardziej agresywne downsampling
        initial_mean_pooling_kernel_size=2,     # DODANE - wymagane
        max_seq_length=400,     # Zmniejsz dla pamięci
    ).to(device)
        
    # Load the model weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    
    model = model.to(device)
    model.eval()
    
    # If audio file is provided, transcribe it
    if args.audio_file:
        if os.path.exists(args.audio_file):
            print(f"Transcribing audio file: {args.audio_file}")
            transcription = transcribe_audio_file(model, tokenizer, args.audio_file, device)
            print(f"Transcription: {transcription}")
        else:
            print(f"Audio file not found: {args.audio_file}")
        return
    
    # Get evaluation dataset
    print("Loading evaluation dataset...")
    eval_dataloader = get_dataset(
        batch_size=args.batch_size,
        num_examples=args.num_examples,
        num_workers=0,  # Avoid multiprocessing issues
    )
    
    # Evaluate model
    print(f"Evaluating model {args.model_id}...")
    eval_results = evaluate_model(model, eval_dataloader, tokenizer, device)
    
    print("\nEvaluation Results:")
    print(f"Word Error Rate: {eval_results['wer']:.4f}")
    print(f"Character Error Rate: {eval_results['cer']:.4f}")
    print(f"Number of samples: {eval_results['num_samples']}")
    
    # Create output directory
    output_dir = f"evaluation_results/{args.model_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'reference': eval_results['references'],
        'prediction': eval_results['predictions'],
        'wer': eval_results['per_sample_wer'],
        'cer': eval_results['per_sample_cer']
    })
    results_df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    
    # Generate plots
    print("Generating plots...")
    plot_error_distribution(eval_results, output_dir)
    plot_confusion_matrix(eval_results, output_dir=output_dir)
    
    # Save examples
    print("Saving example predictions...")
    save_examples(eval_results, output_dir=output_dir)
    
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
