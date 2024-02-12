import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import HubertModel
from transformers import AutoFeatureExtractor, WhisperModel

from model import AssessGradeForDBS
from dataset import MyDataset

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=False,default ="data", help='Path of your DATA/ directory')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='checkpoints/tuika_vowel_new/whsiper', help='Output directory for your trained checkpoints')
    args = parser.parse_args()

    datadir = args.datadir
    ckptdir = args.outdir
    my_checkpoint = args.finetune_from_checkpoint
    
    if not os.path.exists(ckptdir):
        os.system('mkdir -p ' + ckptdir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    trainlist = os.path.join(datadir, 'sets/ready_to_use_tuika_16k_old/train.csv')
    validlist = os.path.join(datadir, 'sets/ready_to_use_tuika_16k_old/dev.csv')

    # ssl_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
    # ssl_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    # ssl_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    # ssl_model = HubertModel.from_pretrained("rinna/japanese-hubert-base")
    # for param in ssl_model.base_model.parameters():
    #     param.requires_grad = False
    SSL_OUT_DIM = 768
    GRBAS_DIM = 1 # multi-task
    ssl_model = HubertModel.from_pretrained("rinna/japanese-hubert-base")
    asr_model = WhisperModel.from_pretrained("openai/whisper-small",output_attentions=True)
    # model.config.decoder_start_token_id
    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
    decoder_input_ids = torch.tensor([[1, 1]]) * asr_model.config.decoder_start_token_id
    decoder_input_ids = decoder_input_ids.to(device)

    trainset = MyDataset(trainlist,feature_extractor)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1, collate_fn=trainset.collate_fn)

    validset = MyDataset(validlist,feature_extractor)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=1, collate_fn=validset.collate_fn)

    net = MosPredictor(ssl_model,asr_model,decoder_input_ids, SSL_OUT_DIM, GRBAS_DIM)
    net = net.to(device)
    # for param in net.parameters():
    #     print(param.requires_grad)
    if my_checkpoint != None:  ## do (further) finetuning
        net.load_state_dict(torch.load(my_checkpoint))
    
    # criterion = nn.L1Loss()
    # weights = [131/83,1,131/88]
    # class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    downsample = nn.Sequential(
            nn.MaxPool1d(400,320)
            # nn.MaxPool1d(10,5),
            # nn.MaxPool1d(3,2),
            # nn.MaxPool1d(3,2),
            # nn.MaxPool1d(3,2),
            # nn.MaxPool1d(3,2),
            # nn.MaxPool1d(2,2),
            # nn.MaxPool1d(2,2)
        )
    # def criterion(est,target):
    #     # print(est)
    #     # print(target)
    #     est = torch.nn.functional.softmax(est,dim=-1)
    #     # print(est)
    #     est = est.squeeze()
    #     target = target.squeeze()
    #     onehot_out = nn.functional.one_hot(target,num_classes=3)
    #     position = torch.argmax(onehot_out)
    #     p = 0
    #     loss = 0
    #     for i,j in zip(est,onehot_out):
    #         cross_loss = -torch.log(1-i)
    #         position_loss = torch.pow(p-position,2)
    #         p = p+1
    #         loss += (cross_loss*position_loss)
    #     # print(loss)
    #     # if est_postion == 0:
    #     #     weight = 3/0.71
    #     # elif est_postion ==1 :
    #     #     weight = 3/1.44
    #     # else:
    #     #     weight = 3/0.85
        
    #     # print(est)
    #     # print(target)
    #     # print(onehot_out)
    #     # print(position)
    #     # print(est_postion,position)
    #     # index = torch.tensor([0,1,2]).to(device =device)
    #     # cross_loss = -torch.log(est[position])
    #     # position_loss = torch.pow(est_postion-position,2)
    #     # print(torch.pow(cross_loss,position_loss))
    #     # print(cross_loss * position_loss)
    #     return loss
    def criterion(est,target):
        # print(est)
        # print(target)
        est = est.squeeze()
        target = target.squeeze()
        onehot_out = nn.functional.one_hot(target,num_classes=3)
        est_postion = torch.argmax(est)
        # if est_postion == 0:
        #     weight = 3/0.71
        # elif est_postion ==1 :
        #     weight = 3/1.44
        # else:
        #     weight = 3/0.85
        position = torch.argmax(onehot_out)
        # print(est)
        # print(target)
        # print(onehot_out)
        # print(position)
        # print(est_postion,position)
        # index = torch.tensor([0,1,2]).to(device =device)
        cross_loss = -torch.log(est[position])
        position_loss = torch.abs(est_postion-position)
        # print(torch.pow(cross_loss,position_loss))
        # print(cross_loss * position_loss)
        return cross_loss * position_loss
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    # optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.8)
    # optimizer = optim.AdamW(net.parameters(), lr=0.001)

    PREV_VAL_LOSS=999999
    orig_patience=20
    patience=orig_patience
    for epoch in range(1,1001):
        STEPS=0
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            asr_mel_features, inputs,mel_specgrams, labels, filenames = data
            # mel_specgram = mel_specgram.to(device)
            asr_mel_features = asr_mel_features.to(device)
            inputs = inputs.to(device)
            mel_specgrams = mel_specgrams.to(device)
            labels = labels.to(device)
            # mask = mask.to(device)
            # target_masks = downsample(mask).squeeze(1)
            # B,T = target_masks.size()
            # labels = labels.long().squeeze(1)
            # print(labels)
            labels = torch.ceil(labels).long().squeeze(1)-1
            # print(labels)
            # labels = labels * target_masks
            # Non_zero = torch.count_nonzero(target_masks,dim=1)

            optimizer.zero_grad()
            outputs = net(asr_mel_features, inputs,mel_specgrams).squeeze(1)
            # print(outputs.size())
            # print(labels.size())
            loss = criterion(outputs, labels)
            # loss = loss/Non_zero
            loss.backward()
            optimizer.step()
            STEPS += 1
            running_loss += loss.item()
        print('EPOCH: ' + str(epoch))
        print('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))
        epoch_val_loss = 0.0
        net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        ## validation
        VALSTEPS=0
        for i, data in enumerate(validloader, 0):
            VALSTEPS+=1

            asr_mel_features, inputs,mel_specgrams, labels, filenames = data
            # mel_specgram = mel_specgram.to(device)
            asr_mel_features = asr_mel_features.to(device)
            inputs = inputs.to(device)
            mel_specgrams = mel_specgrams.to(device)
            labels = labels.to(device)
            # mask = mask.to(device)
            # target_masks = downsample(mask).squeeze(1)
            # print(labels)
            # labels = labels.long().squeeze(1)
            # labels = torch.ceil(labels).long()-1
            labels = torch.ceil(labels).long().squeeze(1)-1
            # print(labels)
            outputs = net(asr_mel_features, inputs,mel_specgrams).squeeze(1)
            # print()
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()
            # if i ==40:
            #     print(labels)
            #     print(outputs)
        avg_val_loss=epoch_val_loss/VALSTEPS    
        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        if avg_val_loss < PREV_VAL_LOSS:
            print('Loss has decreased')
            PREV_VAL_LOSS=avg_val_loss
            PATH = os.path.join(ckptdir, 'ckpt_' + str(epoch)+".pth")
            torch.save(net.state_dict(), PATH)
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                print('loss has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
        
    print('Finished Training')

if __name__ == '__main__':
    main()
