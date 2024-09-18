import torchaudio
from pathlib import Path
from fastai.vision.all import Transform, F
from dataclasses import dataclass

def label_func(f): return f.parent.name


@dataclass
class SpectrogramConfig2:
    f_min: float = 0.0  # Minimum frequency to display
    f_max: float = 10000.0  # Maximum frequency to display
    hop_length: int = 256  # Hop length
    n_fft: int = 2560  # Number of samples for Fourier transform
    n_mels: int = 256  # Number of Mel bins
    pad: int = 0  # Padding
    to_db_scale: bool = True  # Convert to dB scale
    top_db: int = 100  # Top decibel sound
    win_length: int = None  # Window length
    n_mfcc: int = 20  # Number of MFCC features

@dataclass
class AudioConfig2:
    standardize: bool = False  # Standardization flag
    sg_cfg: dataclass = None  # Spectrogram configuration
    duration: int = 4000  # Duration in samples (e.g., 4000 for 4 seconds)
    resample_to: int = 20000  # Resample rate in Hz


class AudioTransform(Transform):
    def __init__(self, config, mode='test'):
        self.config=config
        self.to_db_scale = torchaudio.transforms.AmplitudeToDB(top_db=self.config.sg_cfg.top_db)
        self.spectrogrammer = torchaudio.transforms.MelSpectrogram(
                                                                    sample_rate=self.config.resample_to,
                                                                    n_fft=self.config.sg_cfg.n_fft,
                                                                    hop_length=self.config.sg_cfg.hop_length,
                                                                    n_mels=self.config.sg_cfg.n_mels,
                                                                    f_min=self.config.sg_cfg.f_min,
                                                                    f_max=self.config.sg_cfg.f_max
                                                                )
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=80)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=80)
        self.mode=mode
        
    def encodes(self, fn: Path):
        wave, sr = torchaudio.load(fn)
        wave = wave.mean(dim=0) # reduce to mono
        # resample to 
        wave = torchaudio.functional.resample(wave, sr, self.config.resample_to)

        # pad or truncate to config.duration
        max_len = int(self.config.duration/1000 * self.config.resample_to)

        # print(wave.shape)
        if wave.shape[0] < max_len:
            wave = F.pad(wave, (0, max_len - wave.shape[0]))  # Pad if shorter than max_len
        else:
            wave = wave[:max_len]  # Truncate if longer than max_len

        # print(wave.shape)

        # Generate the MelSpectrogram
        spec = self.spectrogrammer(wave)

        # during training only!
        if self.mode=='train':
            spec = self.time_masking(self.freq_masking(spec))
            
        # Convert the MelSpectrogram to decibel scale if specified
        if self.config.sg_cfg.to_db_scale:
            spec = self.to_db_scale(spec)

        # print('spec',spec.shape)
        spec = spec.unsqueeze(0).expand(3, -1, -1)
        return spec