from seq2seq import *
import numpy as np
from dataload import *
from lm import *
from hred import *
from vhred import *

#hyper-parameters
######################
h_size=3
c_size=3
d_size=3
l_size=1
lr=0.0002
em=torch.ones((9,3))
######################


if __name__ == '__main__':
    #s=seq2seq(em,h_size,d_size,lr)
    s=lm(em,d_size,independence=False, lr=lr)
    #s=hred(em,h_size,c_size,d_size,lr)
    #s=vhred(em,h_size,c_size,d_size,l_size,lr)
    dialogs = [[1,2,3,4,0,5,6,7] for i in range(20)]
    print(dialogs)
    s.run_train(dialogs, dialogs, 20, 6)
