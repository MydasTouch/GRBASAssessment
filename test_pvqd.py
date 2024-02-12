import os
import argparse
import torch
import torch.nn as nn
import fairseq
from torch.utils.data import DataLoader
from model import GRBASPredictor
from dataset import PVQDDataset
import numpy as np

import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig

# from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor
from transformers import WhisperModel
from transformers import HubertModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data', help='Path of your DATA/ directory')
    parser.add_argument('--checkpoint', type=str, default='exp_1/ckpt_87.pth', help='Path to finetuned MOS prediction checkpoint.')
    parser.add_argument('--outfile', type=str, required=False, default='exp_1/result.csv', help='Output filename for your answer.txt file for submission to the CodaLab leaderboard.')
    args = parser.parse_args()
    
    my_checkpoint = args.checkpoint
    datadir = args.datadir
    outfile = args.outfile

    #facebook/hubert-base-ls960
    #rinna/japanese-hubert-base
    ssl_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssl_out_dim = 768
    grbas_dim = 1 # 1 or 5
    multi_indicator = True
    asr_model = WhisperModel.from_pretrained("openai/whisper-small",output_attentions=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
    decoder_input_ids = torch.tensor([[1, 1]]) * asr_model.config.decoder_start_token_id
    decoder_input_ids = decoder_input_ids.to(device)

    model = GRBASPredictor(ssl_model,asr_model, decoder_input_ids,ssl_out_dim,grbas_dim).to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint))

    wavdir = os.path.join(datadir, 'wav')
    validlist = os.path.join(datadir, 'sets/16k_slice_label/test_speech_reg.csv')
    # validlist = os.path.join(datadir, 'sets/16k_slice_label/test_a_reg.csv')
  
    print('Loading data')
    validset = PVQDDataset(wavdir, validlist, feature_extractor, multi_indicator)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=0, collate_fn=validset.collate_fn)

    total_loss = 0.0
    num_steps = 0.0
    predictions = { }  # filename : prediction
    criterion = nn.MSELoss(reduction='sum')
    print('Starting prediction')

    Loss = []
    for i, data in enumerate(validloader, 0):
        asr_mel_features, inputs, mel_specgrams, labels, filenames = data

        asr_mel_features = asr_mel_features.to(device)
        inputs = inputs.to(device)
        mel_specgrams = mel_specgrams.to(device)
        labels = labels.to(device)

        outputs = model(asr_mel_features, inputs, mel_specgrams)
      
        loss = criterion(labels,outputs)
     
        outputs = outputs.cpu().detach().numpy()[0]

        Loss.append(loss.cpu().detach().numpy())
        predictions[filenames[0]] = outputs  ## batch size = 1
       
    print(np.mean(Loss))
    true_G = { }
    validf = open(validlist, 'r')
    for line in validf:
        parts = line.strip().split(',')
        uttID = parts[0]
        if multi_indicator:
            G = [float(i) for i in parts[1:]]
        else:
            G = float(parts[1])
        true_G[uttID] = G

    index = "BL05"
    num = 0
    ans = open(outfile, 'w')

    for k, v in predictions.items():
        # print(k,v,true_G[k])
        # print(v[0],true_G[k][0])
        if index == k.split('.')[0].split('/')[1].split('_')[0]:
            if multi_indicator:
                outl = k.split('.')[0] + ',' +str(num) +','+ str(v[0]) +','+ str(true_G[k][0]) +','+ str(v[1]) +','+ str(true_G[k][1]) +','+ str(v[2]) +','+ str(true_G[k][2])+',' + str(v[3]) +','+ str(true_G[k][3]) +','+ str(v[4]) +','+ str(true_G[k][4])+ '\n'
            else:
                outl = k.split('.')[0] + ',' +str(num) +','+ str(v) +','+ str(true_G[k])+ '\n'
            ans.write(outl)
        else:
            index = k.split('.')[0].split('/')[1].split('_')[0]
            num += 1
            if multi_indicator:
                outl = k.split('.')[0] + ',' +str(num) +','+ str(v[0]) +','+ str(true_G[k][0]) +','+ str(v[1]) +','+ str(true_G[k][1]) +','+ str(v[2]) +','+ str(true_G[k][2])+',' + str(v[3]) +','+ str(true_G[k][3]) +','+ str(v[4]) +','+ str(true_G[k][4])+ '\n'
            else:
                outl = k.split('.')[0] + ',' +str(num) +','+ str(v) +','+ str(true_G[k])+ '\n'
            ans.write(outl)
    ans.close()

if __name__ == '__main__':
    main()
