import random
import torchaudio
import datasets
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing
from pathlib import Path
import sounddevice as sd

def collate_fn(batch):
    # Get max audio length
    max_audio_len = max([item["audio"].shape[0] for item in batch])
    
    # Get max input_ids length if present
    max_ids_len = 0
    has_input_ids = "input_ids" in batch[0]
    if has_input_ids:
        max_ids_len = max([len(item["input_ids"]) for item in batch])
    
    # Pad audio sequences
    audio_tensor = torch.stack(
        [
            F.pad(item["audio"], (0, max_audio_len - item["audio"].shape[0]))
            for item in batch
        ]
    )
    
    output_dict = {
        "audio": audio_tensor,
        "text": [item["text"] for item in batch],
    }

    if has_input_ids:
        # Pad input_ids sequences
        input_ids = torch.stack(
            [
                F.pad(torch.tensor(item["input_ids"]), (0, max_ids_len - len(item["input_ids"])))
                for item in batch
            ]
        )
        output_dict["input_ids"] = input_ids
    return output_dict
        
def get_tokenizer(save_path="tokenizer.json"):
    tokenizer = Tokenizer(models.BPE())
    
    # Add more special tokens for better handling
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<□>"]
    tokenizer.add_special_tokens(special_tokens)
    
    # Add all printable ASCII characters
    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-\" ")
    tokenizer.add_tokens(chars)
    
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # Better than ByteLevel for speech
    tokenizer.decoder = decoders.BPEDecoder()
    
    return tokenizer


class CommonVoiceDataset(Dataset):
    def __init__(
        self,
        common_voice_dataset,
        num_examples=None,
        tokenizer=None,
    ):
        self.dataset = common_voice_dataset
        self.num_examples = (
            min(num_examples, len(common_voice_dataset))
            if num_examples is not None
            else len(common_voice_dataset)
        )
        self.tokenizer = tokenizer
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        waveform = torch.from_numpy(item["audio"]["array"]).float()  # Convert to tensor
        text = item[
            "transcription"
        ].upper()  # Common Voice text is lowercase by default
        if self.tokenizer:
            encoded = self.tokenizer.encode(text)
            return {"audio": waveform, "text": text, "input_ids": encoded.ids}
        
        return {"audio": waveform, "text": text}

def get_dataset(
        batch_size=32,
        num_examples=None,
        num_workers=4,
):
        # Load Common Voice dataset from local file
    dataset = datasets.load_dataset(
        "m-aliabbas/idrak_timit_subsample1",
        split="train",
    )
    tokenizer = get_tokenizer()

        # Create a new dataset instance with the trained tokenizer
    dataset = CommonVoiceDataset(
        dataset,
        tokenizer=tokenizer,
        num_examples=num_examples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return dataloader

if __name__ == "__main__":
    dataloader = get_dataset(
        batch_size=32
    )
    for batch in dataloader:
        audio = batch["audio"]
        input_ids = batch["input_ids"]
        print(audio.shape, input_ids)
        
        breakpoint()
        break