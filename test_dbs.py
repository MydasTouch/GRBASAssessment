import os
import argparse
import torch
import torch.nn as nn
import fairseq
from torch.utils.data import DataLoader
from model import GRBASPredictor
from dataset import DBSDataset
import numpy as np

from transformers import WhisperModel
from transformers import HubertModel
# import random
# random.seed(1895)

def main():
    parser = argparse.ArgumentParser()  # ckpt_18_best_rina_hubert.pth
    parser.add_argument('--datadir', type=str, default='data', help='Path of your DATA/ directory')
    parser.add_argument('--checkpoint', type=str, required=False, default='checkpoints/exp_dbs/ckpt_20.pth')
    parser.add_argument('--outfile', type=str, required=False, default='checkpoints/exp_dbs/result.csv', help='Output filename for your answer.txt file for submission to the CodaLab leaderboard.')
    args = parser.parse_args()
    
    my_checkpoint = args.finetuned_checkpoint
    outfile = args.outfile
    datadir = args.datadir

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssl_out_dim  = 768
    grbas_dim  = 3
    multi_indicator = False

    # ssl module
    ssl_model = HubertModel.from_pretrained("rinna/japanese-hubert-base")

    # asr module
    asr_model = WhisperModel.from_pretrained("openai/whisper-small",output_attentions=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
    decoder_input_ids = torch.tensor([[1, 1]]) * asr_model.config.decoder_start_token_id
    decoder_input_ids = decoder_input_ids.to(device)
    
    model = GRBASPredictor(ssl_model, asr_model, decoder_input_ids,ssl_out_dim,grbas_dim).to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint))
  
    for name, param in model.named_parameters():
        if name == "ssl_weight":
            print(param)
  
    validlist = os.path.join('data','sets/ready_to_use_tuika_16k_old/test.csv')

    print('Loading data')
    
    validset = DBSDataset(validlist,feature_extractor, multi_indicator)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=1, collate_fn=validset.collate_fn)

    total_loss = 0.0
    num_steps = 0.0
    predictions = { }  # filename : prediction
    criterion = nn.L1Loss()

    for i, data in enumerate(validloader, 0):
        asr_mel_features, inputs,mel_specgrams, labels, filenames = data
      
        asr_mel_features = asr_mel_features.to(device)
        inputs = inputs.to(device)
        mel_specgrams = mel_specgrams.to(device)
        labels = labels.to(device)

        outputs = model(asr_mel_features, inputs,mel_specgrams).squeeze(1)
        outputs_ = torch.argmax(outputs,dim=-1)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        outputs_ = outputs_.cpu().detach().numpy()[0]
        predictions[filenames[0]] = outputs_  ## batch size = 1
    true_G = { }
    validf = open(validlist, 'r')
    for line in validf:
        parts = line.strip().split(',')
        uttID = parts[0]
        if multi_indicator:
            G = [float(i) for i in parts[5:10]]
        else:
            G = float(parts[5])
        G  = np.ceil(G)-1
        true_G[uttID] =G

    ans = open(outfile, 'a+')
    for k, v in predictions.items():
        outl = k.split('.')[0]  +','+ str(v) +','+ str(true_G[k])+ '\n'
        ans.write(outl)
    ans.close()

if __name__ == '__main__':
    main()
