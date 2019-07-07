import argparse
import numpy as np
import os
import configparser
import torch
import torch.nn as nn
from torch.autograd import Variable
# from torchvision import datasets, transforms
from fat2019_dataset  import FATDataset
import model4prune
import pickle as pkl

def prune_convWeights(model_path,class_num,percent):
    model = model4prune.CNNModelBasic(class_num)
    print ('loading model from {}...'.format(model_path))
    model.load_state_dict(torch.load(model_path))
    # print (model)

    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * percent)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.AvgPool2d):
            cfg.append('A')

    pruned_ratio = pruned/total

    print('Pre-processing Successful! pruned ratio:{}'.format(pruned_ratio))
    return model, cfg, cfg_mask

def get_newmodel(model,class_num,cfg,cfg_mask):
    newmodel = model4prune.CNNModelBasic(class_num,cfg)
    # print (newmodel)
    mask_idx = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[mask_idx]
    reach_Linear = False
    for m0,m1 in zip(model.modules(),newmodel.modules()):
        if reach_Linear:
            # pruning only done on CNN layer + the first FC layer
            break
        if isinstance(m0,nn.BatchNorm2d):
            print ('copying bn weight...')
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.numpy())))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()

            mask_idx += 1
            start_mask = end_mask.clone()
            if mask_idx < len(cfg_mask):
                end_mask = cfg_mask[mask_idx]

        elif isinstance(m0,nn.Conv2d):
            print ('copying conv2d weight...')
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.numpy())))
            w1 = m0.weight.data[:,idx0.tolist(),:,:].clone()
            w1 = w1[idx1.tolist(),:,:,:].clone()
            m1.weight.data = w1.clone()

        elif isinstance(m0,nn.Linear):
            print ('copying linear weight...')
            # this should be the first linear after the last bn layer
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.numpy())))
            m1.weight.data = m0.weight.data[:,idx0.tolist()].clone()
            m1.bias.data = m0.bias.data.clone()
            reach_Linear = True
    print ('model pruning finished!')
    return newmodel



if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     print ('use cuda ...')
    #     device = 'cuda'
    # else:
    #     print ('use cpu...')
    device = 'cpu'

    config=configparser.ConfigParser()
    config.read('./config.ini')

    traindatadir = config.get('DataPath','train_data_path')
    valdatadir = config.get('DataPath','val_data_path')
    traindatacsv = config.get('DataPath','train_csv_path')
    valdatacsv  = config.get('DataPath','val_csv_path')
    prune_ratio = float(config.get('Parameters','prune_ratio')) # percent of channels to be pruned
    print ('initialize dataset...')
    voiceDataset = FATDataset(traindatadir, valdatadir, traindatacsv,valdatacsv,batch_size=8)
    class_num = voiceDataset.get_class_num()
    model,cfg,cfg_mask = prune_convWeights(os.path.join(config.get('DataPath','checkpoint_dir'),config.get('SaveModel','model_before_prune')),\
                                            class_num,prune_ratio)
    print (cfg)
    newmodel = get_newmodel(model,class_num,cfg,cfg_mask)
    print (newmodel)
    torch.save(newmodel.state_dict(), os.path.join(config.get('DataPath','checkpoint_dir'),config.get('SaveModel','pruned_model')))
    pkl.dump(cfg,open(os.path.join(config.get('DataPath','checkpoint_dir'),config.get('SaveModel','pruned_cfg')),'wb'))
