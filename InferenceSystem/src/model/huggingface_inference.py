from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torchaudio
import torch

import pandas as pd
from pathlib import Path
from numpy import floor
import os

import time



class HFDataset(torch.utils.data.Dataset):
    def __init__(self, data, feature_extractor, max_length=15, fs=32000, mono=True):
        self.data = data
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.fs = fs

        self.mono = mono # only one channel

        self.resampler = {}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        wav_file_path, start_time, end_time = self.data[idx]

        # Load the audio file
        try:
            data, r = torchaudio.load(wav_file_path) #frame_offset=frame_offset, num_frames=num_frames
        except:
            
            time.sleep(2)
            assert os.path.exists(wav_file_path), print(wav_file_path, 'does not exist')
            data, r = torchaudio.load(wav_file_path) #frame_offset=frame_offset, num_frames=num_frames

            try:
                data = torchaudio.functional.highpass_biquad(data, r,  1000,)
            except:
                print('failed',wav_file_path)
                print(data.shape)
                raise

        # check if any data is nan
        if torch.isnan(data).any():
            raise Exception(wav_file_path)
        
        if r!= self.fs:
            # resample
            if r not in self.resampler.keys():
                self.resampler.update({r:torchaudio.transforms.Resample(r, self.fs)})
            try:data = self.resampler[r](data)
            except:
                print(data.shape)
                print(wav_file_path)
                data = self.resampler[r](data)

        if torch.isnan(data).any():
            raise Exception(wav_file_path)
        
        if not(self.mono):
            data=data.expand(2,-1) # because it is mono
        elif data.shape[0]==2:
            # check if one side is all nans?
            if torch.sum(data[0])==0:
                data=data[1]
            elif torch.sum(data[1])==0:
                print("it was all 0")
                raise
                data=data[0]
            else:
                data=data.mean(0, keepdim=True)


        # check if any data is nan
        if torch.isnan(data).any():
            raise Exception(wav_file_path)
        
        if len(data.shape)==1:
            if self.mono:
                data =data.view(1, -1)
            else:
                data = data.view(2, -1)

        # Get the correct index

        if end_time is not None:
            data = data[:, int(start_time*self.fs):int(end_time*self.fs)]
        else:
            data = data[:, int(start_time*self.fs):]

        # print(len(audio3), len(audio2))
        # assert len(audio3) == len(audio2), f"audio3 and audio2 must be the same length, {len(audio3)}!={len(audio2)}"


        # print('unfold',audio.shape)
        data = [data.squeeze().cpu().data.numpy()]

        pad_max = 32000*15
        # we have to lie and tell the feature_extractor that the data is sampled at 16000, because that is what it was pretrained on, however, it is rpobust to the sampling rate fo 32000 due to fine-tuning.
        data = self.feature_extractor(data, sampling_rate = 16000, padding='max_length', max_length=int(pad_max), return_tensors='pt')

        try:
            data['input_values'] = data['input_values'].squeeze(0)
        except:
            # it is called input_features for whisper
            data['input_features'] = data['input_features'].squeeze(0)

        return data




class HuggingfaceModel():
    """
    This is a wrapper for huggingface models so that they return json objects and consider the same configs as other implementations
    """
    def __init__(self, model_path=None, model_name=None, threshold=0.5, min_num_positive_calls_threshold=3):

        # Load the model
        load_path = model_path if model_path is not None else model_name
        self.model = AutoModelForAudioClassification.from_pretrained(load_path)
        self.tokenizer = AutoFeatureExtractor.from_pretrained(load_path)

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        
        self.threshold = threshold
        self.min_num_positive_calls_threshold = min_num_positive_calls_threshold

    def predict(self, wav_file_path):
        '''
        Function which generates local predictions using wavefile
        '''

        # This model operates on 15-second-long audio files.        

        # infer clip length
        metadata = torchaudio.info(wav_file_path)
        max_length = metadata.num_frames / metadata.sample_rate

        # create a list of data with [filename, start_time, end_time]
        start_times = list(range(0, int(max_length), 15))
        end_times = [s+15 for s in start_times]
        end_times[-1] = None # if none we will go until the end of the file
        data = list(zip([wav_file_path]*len(start_times), start_times, end_times)) 

        dataset = HFDataset(data, self.tokenizer, max_length=15, fs = 32000) # huggingface needs the tokenizer instead of spectrograms
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) # you could increaase num_workers if it is a multi-cpu machine

        # Scoring each 15 sec clip
        predictions = []
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(**batch)
            predictions.append(torch.softmax(output.logits, dim=1)[0][1].cpu().data.numpy()) # 1 prediction per 15 second clip


        # Aggregating predictions

        filenames, start_time, end_time = zip(*data)

        # Creating a DataFrame
        prediction = pd.DataFrame({'wav_filename': filenames, 'start_time_s':start_time, 'end_time_s':end_time, 'confidence': predictions})

        prediction.loc[:, 'end_time_s'] = prediction['end_time_s'].fillna(max_length)
        prediction.loc[:, 'duration_s'] = prediction['end_time_s'] - prediction['start_time_s']

        # skipping this rolling window
        # # Rolling Window (to average at per second level)
        # submission = pd.DataFrame(
        #         {
        #             'wav_filename': Path(wav_file_path).name,
        #             'duration_s': 1.0,
        #             'confidence': list(prediction.rolling(2)['confidence'].mean().values)
        #         }
        #     ).reset_index().rename(columns={'index': 'start_time_s'})

        # # Updating first row
        # submission.loc[0, 'confidence'] = prediction.confidence[0]

        # # Adding lastrow
        # lastLine = pd.DataFrame({
        #     'wav_filename': Path(wav_file_path).name,
        #     'start_time_s': [submission.start_time_s.max()+1],
        #     'duration_s': 1.0,
        #     'confidence': [prediction.confidence[prediction.shape[0]-1]]
        #     })
        
        # submission = submission.append(lastLine, ignore_index=True)
        # submission = submission[['wav_filename', 'start_time_s', 'duration_s', 'confidence']]


        # initialize output JSON
        result_json = {}
        result_json = dict(
            submission=prediction[['wav_filename', 'start_time_s','duration_s', 'confidence']].to_dict(orient='records'),
            local_predictions=list((prediction['confidence'] > self.threshold).astype(int)),
            local_confidences=list(prediction['confidence'])
        )

        result_json['global_prediction'] = int(sum(result_json["local_predictions"]) > self.min_num_positive_calls_threshold)
        result_json['global_confidence'] = prediction.loc[(prediction['confidence'] > self.threshold), 'confidence'].mean()*100
        if pd.isnull(result_json["global_confidence"]):
            result_json["global_confidence"] = 0

        return result_json
