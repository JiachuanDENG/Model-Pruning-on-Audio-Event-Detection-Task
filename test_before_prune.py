from fat2019_dataset  import FATDataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tqdm import tqdm
import collections
import numpy as np
import matplotlib.pyplot as plt
import math
import utils
import glob
from random import shuffle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.autograd as autograd
import logging
from time import gmtime, strftime
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import oneSampleOutput
import configparser
import sys
import model4prune
from thop import profile
import timeit

config=configparser.ConfigParser()
config.read('./config.ini')


currenttime = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
checkpoint_dir = './checkpoint'
if not os.path.exists(checkpoint_dir):
    os.system('mkdir {}'.format(checkpoint_dir))


logger = logging.getLogger('freesound_kaggle')
hdlr = logging.FileHandler('./log/test_before_prune_{}.log'.format(currenttime))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)
hdlr.setLevel(logging.INFO)




def test(traindatadir, testdatadir, traindatacsv,testdatacsv,device,model_path=''):
    """
    test data is loaded in same format as validation data
    """
    print ('initialize dataset...')
    voiceDataset = FATDataset(traindatadir, testdatadir, traindatacsv,testdatacsv,batch_size=8)
    # print (voiceDataset.get_class_num())
    print ('create model ... ')

    # cnnmodel = models.CNNModelv2(voiceDataset.get_class_num()).to(device)
    cnnmodel = model4prune.CNNModelBasic(voiceDataset.get_class_num()).to(device)
    #loading trained model
    print ('loading model from {}...'.format(model_path))
    cnnmodel.load_state_dict(torch.load(model_path))

    # testing
    cnnmodel.eval()
    test_batches_num = voiceDataset.get_numof_batch(False)
    test_preds = np.array([]).reshape(0,voiceDataset.get_class_num())
    test_labels = np.array([]).reshape(0,voiceDataset.get_class_num())

    # calculate flops and params
    flop_test_data,_,__ = voiceDataset.get_data(0,False)
    flop_test_data = flop_test_data[0:1,:,:,:].to(device)
    flops,params = profile(cnnmodel,inputs = (flop_test_data,))


    eval_start = timeit.default_timer()
    for tbidx  in tqdm(range(test_batches_num)):
        # print ('generating validation fingerprint ... ')
        test_data,test_samplenumbatch,test_label = voiceDataset.get_data(tbidx,False)

        pred = oneSampleOutput(cnnmodel(test_data.to(device)).detach(),test_samplenumbatch).to(device)
        test_preds = np.vstack((test_preds,pred.cpu().numpy()))

        test_labels = np.vstack((test_labels,test_label.cpu().numpy()))
    eval_stop = timeit.default_timer()

    score, weight = utils.calculate_per_class_lwlrap(test_labels, test_preds)
    lwlrap = (score * weight).sum()
    msg = '[TESTING]  lwlrap:{}, flops:{}, params:{}, running time:{}'.format(lwlrap,flops,params,eval_stop-eval_start)

    print (msg)
    logger.info(msg)

if __name__ == '__main__':
    if False:
    # if torch.cuda.is_available():
        print ('use cuda ...')
        device = 'cuda'
    else:
        print ('use cpu...')
        device = 'cpu'

    model_path = os.path.join(config.get('DataPath','checkpoint_dir'),config.get('SaveModel','model_before_prune'))

    # stage final: testing
    # datapath, csvpath below can be any data, we don't use them in this testing stage actually, the reason we need to define them is
    # just because reusing the dataset class defined need these things
    datapath = config.get('DataPath','train_data_path')
    csvpath  = config.get('DataPath','train_csv_path')

    # test_datapath, test_csv_path are data we are actually using in this stage
    test_datapath = config.get('DataPath','test_data_path')
    test_csv_path = config.get('DataPath','test_csv_path')



    test(datapath,test_datapath,csvpath,test_csv_path,device,model_path)
