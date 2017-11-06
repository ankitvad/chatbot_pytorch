from seq2seq import *
import numpy as np
from dataload import *
from lm import *
from hred import *
from vhred import *
import pickle

#hyper-parameters
######################
h_size=512#encoder hidden size
c_size=1024#context hidden size
d_size=512#decoder hidden size
l_size=128#latent variable size
b_size=128#batch size
lr=0.0002#learning rate
embed='./dataset/switchboard/embedding.mat'#pre-trained embedding
train_pkl = './dataset/switchboard/train.pkl'#training set
valid_pkl = './dataset/switchboard/valid.pkl'#validation set
test_pkl = './dataset/switchboard/test.pkl'#validation set
check_dir = './hred'#check point file
######################

torch.set_default_tensor_type('torch.cuda.FloatTensor')
if __name__ == '__main__':
    em =torch.from_numpy(pickle.load(open(embed, 'rb'), encoding='latin1'))
    #s=seq2seq(em,h_size,d_size,lr)
    #s=lm(em,d_size,independence=False, lr=lr)
    s=hred(em,h_size,c_size,d_size,lr)
    #s=vhred(em,h_size,c_size,d_size,l_size,lr)
    s=s.cuda()
    train_dialogs = pickle.load(open(train_pkl,'rb'))
    valid_dialogs = pickle.load(open(valid_pkl,'rb'))
    test_dialogs = pickle.load(open(test_pkl,'rb'))
    #trained = dialogdata(train_dialogs)
    s.run_train(train_dialogs, valid_dialogs, num_epochs=50, b_size=b_size, check_dir = check_dir)
