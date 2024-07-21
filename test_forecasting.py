import torch
import torch.nn as nn
# from utils_dir.Similarity import get_similarity
import numpy as np
import os
import yaml
import models
import argparse
import datetime

from models.BiaTCGNet.BiaTCGNet import Model
from data.GenerateDataset import loaddataset

import matplotlib.pyplot as plt
node_number=207
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--task',default='prediction', type=str)
parser.add_argument('--hid_size', type=int)
parser.add_argument('--impute_weight', type=float)
parser.add_argument('--label_weight', type=float)
parser.add_argument("--adj-threshold", type=float, default=0.1)
parser.add_argument('--dataset',default='Metr')
parser.add_argument('--val_ratio',default=0.2)
parser.add_argument('--test_ratio',default=0.2)
parser.add_argument('--column_wise',default=False)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument("--model-name", type=str, default='spin')
parser.add_argument("--dataset-name", type=str, default='air36'
                                                        '')


parser.add_argument('--fc_dropout', default=0.2, type=float)
parser.add_argument('--head_dropout', default=0, type=float)
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--patch_len', type=int, default=8, help='patch length')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--kernel_set', type=list, default=[2,3,6,7], help='kernel set')
##############transformer config############################
parser.add_argument('--enc_in', type=int, default=node_number, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=node_number, help='decoder input size')
parser.add_argument('--c_out', type=int, default=node_number, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--num_nodes', type=int, default=node_number, help='dimension of fcn')
parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')
#######################AGCRN##########################
parser.add_argument('--input_dim', default=1, type=int)
parser.add_argument('--output_dim', default=1, type=int)
parser.add_argument('--embed_dim', default=512, type=int)
parser.add_argument('--rnn_units', default=64, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--cheb_k', default=2, type=int)
parser.add_argument('--default_graph', type=bool, default=True)

#############GTS##################################
parser.add_argument('--temperature', default=0.5, type=float, help='temperature value for gumbel-softmax.')

parser.add_argument("--config_filename", type=str, default='./models/GTS/para_Metr.yaml')
#####################################################
parser.add_argument("--config", type=str, default='imputation/spin.yaml')
parser.add_argument('--output_attention', type=bool, default=False)
# Splitting/aggregation params
parser.add_argument('--val-len', type=float, default=0.2)
parser.add_argument('--test-len', type=float, default=0.2)
parser.add_argument('--mask_ratio',type=float,default=0.2)
# Training params
parser.add_argument('--lr', type=float, default=0.001)  #0.001
# parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--patience', type=int, default=40)
parser.add_argument('--l2-reg', type=float, default=0.)
# parser.add_argument('--batches-epoch', type=int, default=300)
parser.add_argument('--batch-inference', type=int, default=32)
parser.add_argument('--split-batch-in', type=int, default=1)
parser.add_argument('--grad-clip-val', type=float, default=5.)
parser.add_argument('--loss-fn', type=str, default='l1_loss')
parser.add_argument('--lr-scheduler', type=str, default=None)
parser.add_argument('--seq_len',default=24,type=int)
# parser.add_argument('--history_len',default=24,type=int)
parser.add_argument('--label_len',default=12,type=int)
parser.add_argument('--pred_len',default=24,type=int)
parser.add_argument('--horizon',default=24,type=int)
parser.add_argument('--delay',default=0,type=int)
parser.add_argument('--stride',default=1,type=int)
parser.add_argument('--window_lag',default=1,type=int)
parser.add_argument('--horizon_lag',default=1,type=int)

args = parser.parse_args()
criteron=nn.L1Loss().cuda()
if(args.dataset=='Metr'):
    node_number=207
    args.num_nodes=207
    args.enc_in=207
    args.dec_in=207
    args.c_out=207
elif(args.dataset=='PEMS'):
    node_number=325
    args.num_nodes=325
    args.enc_in = 325
    args.dec_in = 325
    args.c_out = 325
elif(args.dataset=='ETTh1'):
    node_number=7
    args.num_nodes=7
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
elif(args.dataset=='Elec'):
    node_number=321
    args.num_nodes=321
    args.enc_in = 321
    args.dec_in = 321
    args.c_out = 321
elif(args.dataset=='BeijingAir'):
    node_number=36
    args.num_nodes=36
    args.enc_in = 36
    args.dec_in = 36
    args.c_out = 36


def MAPE_np(pred, true, mask_value=0):
    if mask_value != None:
        mask = np.where(np.abs(true) > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), (true))))*100


def RMSE_np(pred, true, mask_value=0):
    if mask_value != None:
        mask = np.where(np.abs(true) > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE
def test(model):
    loss = 0.0
    labels = []
    preds = []
    exp_name = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    exp_name = f"{exp_name}_{args.seed}"
    logdir = os.path.join('./log_dir', args.dataset_name,
                          args.model_name, exp_name)
    # save config for logging
    os.makedirs(logdir, exist_ok=True)

    train_dataloader,val_dataloader, test_dataloader, scaler = loaddataset(args.seq_len,args.pred_len,args.mask_ratio,args.dataset)
    model.eval()
    k=0
    with torch.no_grad():
        for i, (x, y, mask, target_mask) in enumerate(test_dataloader):
            x, y, mask,target_mask = x.cuda(), y.cuda(), mask.cuda(), target_mask.cuda()
            x_hat=model(x,mask,k)
            k=k+1
            x_hat=scaler.inverse_transform(x_hat)

            y=scaler.inverse_transform(y)
            preds.append(x_hat.squeeze())
            labels.append(y.squeeze())
            losses = torch.sum(torch.abs(x_hat-y)*(target_mask))/torch.sum(target_mask)
            # print('losses:',losses)
            loss+=losses

        labels = torch.cat(labels,dim=0).cpu().numpy()
        preds = torch.cat(preds,dim=0).cpu().numpy()

        print('mask loss:',loss/len(test_dataloader))
        loss=np.mean(np.abs(labels.squeeze()-preds.squeeze()))
        RMSE = RMSE_np(preds.squeeze(),labels.squeeze())
        MAPE=MAPE_np(preds.squeeze(),labels.squeeze())
        print('loss,RMSE,MAPE %.2f & %.2f & %.2f' % (loss,RMSE,MAPE))

    return loss

def run():



    model = Model(True, True, 2, node_number,args.kernel_set,
                  'cuda:0', predefined_A=None,
                  dropout=0.3, subgraph_size=5,
                  node_dim=3,
                  dilation_exponential=1,
                  conv_channels=8, residual_channels=8,
                  skip_channels=16, end_channels=32,
                  seq_length=args.seq_len, in_dim=1, out_len=args.pred_len, out_dim=1,
                  layers=2, propalpha=0.05, tanhalpha=3, layer_norm_affline=True).cuda()

    model.load_state_dict(torch.load('./output_BiaTCGNet_'+args.dataset+'_miss'+str(args.mask_ratio)+'_'+args.task+'/best.pth'))
    loss=test(model)

    print('loss:',loss)

if __name__ == '__main__':
    run()

