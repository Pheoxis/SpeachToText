import torch
import torchaudio.transforms as T

class AudioAugmentation:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.time_masking = T.TimeMasking(time_mask_param=80)
        self.freq_masking = T.FreqMasking(freq_mask_param=27)
        
    def __call__(self, audio):
        # Add noise
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(audio) * 0.005
            audio = audio + noise
        
        # Speed perturbation
        if torch.rand(1) < 0.3:
            speed_factor = torch.uniform(0.9, 1.1, (1,)).item()
            audio = T.Speed(self.sample_rate, speed_factor)(audio)
        
        return audio
