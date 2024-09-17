# from fastai.basic_train import load_learner
from fastai.vision.all import *
import pandas as pd
from pydub import AudioSegment
from librosa import get_duration
from pathlib import Path
from numpy import floor
# from audio.data import AudioConfig, SpectrogramConfig, AudioList
from fastai.data.all import *
import torchaudio
import os
import shutil
from dataclasses import dataclass



def load_model(mPath, mName="stg2-rn18.pkl"):
    return load_learner(os.path.join(mPath, mName))


def get_wave_file(wav_file):
    '''
    Function to load a wav file
    '''
    return AudioSegment.from_wav(wav_file)


def export_wave_file(audio, begin, end, dest):
    '''
    Function to extract a smaller wav file based start and end duration information
    '''
    sub_audio = audio[begin * 1000:end * 1000]
    sub_audio.export(dest, format="wav")


def extract_segments(audioPath, sampleDict, destnPath, suffix):
    '''
    Function to exctact segments given a audio path folder and proposal segments
    '''
    # Listing the local audio files
    local_audio_files = str(audioPath) + '/'
    for wav_file in sampleDict.keys():
        audio_file = get_wave_file(local_audio_files + wav_file)
        for begin_time, end_time in sampleDict[wav_file]:
            output_file_name = wav_file.lower().replace(
                '.wav', '') + '_' + str(begin_time) + '_' + str(
                    end_time) + suffix + '.wav'
            output_file_path = destnPath + output_file_name
            export_wave_file(audio_file, begin_time,
                             end_time, output_file_path)





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


# Integrate with fastai's DataBlock (customized for your use case)
def label_func(f): 
    return f.parent.name



class FastAIModel():
    def __init__(self, model_path, model_name="stg2-rn18.pkl", threshold=0.5, global_aggregation_percentile_threshold=3):
        # self.audio_tranform = AudioTransform()
        self.model = load_model(model_path, model_name)
        self.threshold = threshold
        self.global_aggregation_percentile_threshold = global_aggregation_percentile_threshold

    def predict(self, wav_file_path):
        '''
        Function which generates local predictions using wavefile
        '''

        # Creates local directory to save 2 second clips
        local_dir = "./fastai_dir/"
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # infer clip length
        max_length = get_duration(filename=wav_file_path)
        max_length = 60
        # Generating 2 sec proposal with 1 sec hop length
        twoSecList = []
        for i in range(int(floor(max_length)-1)):
            twoSecList.append([i, i+2])

        # Creating a proposal dictionary
        two_sec_dict = {}
        two_sec_dict[Path(wav_file_path).name] = twoSecList

        # local directory
        extract_segments(
            str(Path(wav_file_path).parent),
            two_sec_dict,
            local_dir,
            ""
        )

        spec_cfg = SpectrogramConfig2(
                                 f_min=0.0,  # Minimum frequency to Display
                                 f_max=10000,  # Maximum Frequency to Display
                                 hop_length=256,
                                 n_fft=2560,  # Number of Samples for Fourier
                                 n_mels=256,  # Mel bins
                                 pad=0,
                                 to_db_scale=True,  # Converting to DB sclae
                                 top_db=100,  # Top decible sound
                                 win_length=None,
                                 n_mfcc=20)

        # Definining Audio config needed to create on the fly mel spectograms
        config = AudioConfig2(standardize=False,
                             sg_cfg=spec_cfg,
                             duration=4000, # 4 sec padding or snip
                             resample_to = 20000, # Every sample at 20000 frequency
                             )

        # Creating a Audio DataLoader
        test_data_folder = Path(local_dir)

        audio_transform = AudioTransform(config, mode='test')


        # Define your DataBlock
        audio_block = DataBlock(
            blocks=(TransformBlock, CategoryBlock),
            get_items=get_files,
            get_x=audio_transform,
            get_y=label_func,
            splitter=RandomSplitter(),
            item_tfms=[],
            batch_tfms=[]
        )
        
        # Create DataLoaders
        test_dls = audio_block.dataloaders(test_data_folder, bs=32)
        
        # tfms = None
        # test = AudioList.from_folder(
        #     test_data_folder, config=config).split_none().label_empty()
        # testdb = test.transform(tfms).databunch(bs=32)

        # Scoring each 2 sec clip
        predictions = []
        pathList = []
        for pathname, item in zip(test_dls.items, [item[0] for item in test_dls.train_ds]):    
            predictions.append(self.model.predict(item)[2][1].cpu().data.numpy().tolist())
            pathList.append(str(pathname))
        # for item in testdb.x:
        #     predictions.append(self.model.predict(item)[2][1])
        #     pathList.append(str(item.path))

        # clean folder
        shutil.rmtree(local_dir)

        # Aggregating predictions

        # Creating a DataFrame
        prediction = pd.DataFrame({'FilePath': pathList, 'pred': predictions})

        # Converting prediction to float
        prediction['pred'] = prediction.pred.astype(float)

        # Extracting filename
        prediction['FileName'] = prediction.FilePath.apply(
            lambda x: os.path.basename(x).split("-")[0])

        # Extracting Starting time from file name
        prediction['startTime'] = prediction.FileName.apply(
            lambda x: int(x.split('__')[1].split('.')[0].split('_')[0]))

        # Sorting the file based on startTime
        prediction = prediction.sort_values(
            ['startTime']).reset_index(drop=True)

        # Rolling Window (to average at per second level)
        submission = pd.DataFrame({'pred': list(prediction.rolling(
            2)['pred'].mean().values)}).reset_index().rename(columns={'index': 'StartTime'})

        # Updating first row
        submission.loc[0, 'pred'] = prediction.pred[0]

        # Adding lastrow
        lastLine = pd.DataFrame({'StartTime': [submission.StartTime.max(
        )+1], 'pred': [prediction.pred[prediction.shape[0]-1]]})
        # submission = submission.append(lastLine, ignore_index=True)
        submission = pd.concat((submission, lastLine), ignore_index=True)


        # initialize output JSON
        result_json = {}
        result_json["local_predictions"] = list(
            (submission['pred'] > 0.5).astype(int))
        result_json["local_confidences"] = list(submission['pred'])
        result_json["global_predictions"] = int(sum(
            result_json["local_predictions"]) > self.global_aggregation_percentile_threshold)
        result_json["global_confidence"] = submission.loc[(
            submission['pred'] > 0.5), 'pred'].mean()

        return result_json
