import os, json, glob
import torch
import numpy as np
import src.params as params
import argparse

from src.model import get_model_or_checkpoint
from scipy.io import wavfile
from collections import defaultdict
from src.dataloader import AudioFileWindower 
from pathlib import Path
# inputs: model, mean invstd, input_wav, output_chunk_dir, output_json_dir

# iterate through windows and save audio chunks
def inference_and_write_chunks(
    audio_file_windower,output_chunk_dir,predictions_dir,model_path,
    chunk_duration=params.INFERENCE_CHUNK_S):

    # initialize model from checkpoint
    model, _ = get_model_or_checkpoint(params.MODEL_NAME,model_path,use_cuda=True)
    blob_root = "https://podcaststorage.blob.core.windows.net/orcasoundlabchunked"

    # iterate through windows in dataloader, store current chunk windows and length
    curr_chunk, curr_chunk_duration, curr_chunk_json = [], 0, {}
    file_chunk_counts = defaultdict(int)

    for i in range(len(audio_file_windower)):

        # first get an audio window 
        audio_file_windower.get_mode = 'audio'
        audio_window, _ = audio_file_windower[i]
        _, _, _, af = audio_file_windower.windows[i]
        window_s = audio_file_windower.window_s
        # details for the current chunk
        postfix = '_'+format(file_chunk_counts[af.name],'04x')
        chunk_file_name = (Path(af.name).stem+postfix+'.wav')
        absolute_time = Path(af.name).stem  # NOTE: assumes orcasound data
        output_file_path = output_chunk_dir / chunk_file_name 
        blob_uri = blob_root+'/'+chunk_file_name

        # add window to current chunk
        curr_chunk_duration += window_s
        curr_chunk.append(audio_window)

        # get a mel spec for the window 
        audio_file_windower.get_mode = 'mel_spec'
        mel_spec_window, _ = audio_file_windower[i]
        # run inference on window
        input_data = torch.from_numpy(mel_spec_window).float().unsqueeze(0).unsqueeze(0)
        pred, embed = model(input_data)
        posterior = np.exp(pred.detach().cpu().numpy())
        pred_id = torch.argmax(pred, dim=1).item()
        confidence = round(float(posterior[0,1]),3)

        # trigger and update JSON for current chunk if positive prediction 
        if confidence>0.9:  
            if len(curr_chunk_json)==0:
                # add the header fields (uri, absolute_time, source_guid, annotations)
                curr_chunk_json["uri"] = blob_uri
                curr_chunk_json["absolute_time"] = absolute_time 
                curr_chunk_json["source_guid"] = "rpi_orcasound_lab" 
                curr_chunk_json["annotations"] = [] 
            start_s, duration_s = curr_chunk_duration-window_s, window_s 
            curr_chunk_json["annotations"].append(
                    {
                        "start_time_s":start_s,
                        "duration_s":duration_s,
                        "confidence":confidence
                    }
                )
            print("Positive prediction at {:.2f}, Confidence {:.3f}!".format(start_s,confidence))

        # if exceeds chunk_duration, write chunk and JSON and reset
        if curr_chunk_duration > chunk_duration:
            # write out the chunk and reset
            print("Writing out chunk:",output_file_path.name)
            wavfile.write(output_file_path,af.sr,np.concatenate(curr_chunk))

            # if there are predictions, write JSON to file and clear
            if len(curr_chunk_json)!=0:
                with open(predictions_dir/(Path(chunk_file_name).stem+".json"),'w') as fp:
                    json.dump(curr_chunk_json,fp)
                curr_chunk_json = {}

            # clearing up this chunk
            curr_chunk, curr_chunk_duration = [], 0.
            file_chunk_counts[af.name] += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-wavMasterPath', default=None, type=str, required=True)
    parser.add_argument('-modelPath', default='AudioSet_fc_all', type=str, required=True)
    parser.add_argument('-outputChunkDir', default=None, type=str, required=True)
    parser.add_argument('-outputPredsDir', default=None, type=str, required=True)

    args = parser.parse_args()
    wavmaster_path = Path(args.wavMasterPath)
    output_chunk_dir = Path(args.outputChunkDir)
    model_path = Path(args.modelPath)
    preds_dir = Path(args.outputPredsDir)
    os.makedirs(output_chunk_dir,exist_ok=True)
    os.makedirs(preds_dir,exist_ok=True)

    mean, invstd = model_path/params.MEAN_FILE, model_path/params.INVSTD_FILE 
    wav_file_paths = [ Path(p) for p in glob.glob(args.wavMasterPath+"/*.wav") ]
    # wav_file_paths = [ wavmaster_path/p for p in os.listdir(wavmaster_path) ]
    windower = AudioFileWindower(wav_file_paths,mean=mean,invstd=invstd)
    inference_and_write_chunks(windower,output_chunk_dir,preds_dir,model_path)
