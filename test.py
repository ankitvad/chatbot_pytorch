import pickle
import numpy as np  
from lm import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')
EOT = 2
h_size=512#encoder hidden size
c_size=1024#context hidden size
d_size=512#decoder hidden size
l_size=128#latent variable size
b_size=128#batch size
lr=0.0002#learning rate
embed='./dataset/switchboard/embedding.mat'#pre-trained embedding
model_path = './lm/epoch29'
#torch.set_default_tensor_type('torch.cuda.floattensor')
em = torch.from_numpy(pickle.load(open(embed, 'rb'), encoding='latin1'))
 
#s = lm(em,d_size,False,lr)
s = torch.load(model_path)

with open("dataset/switchboard/dict.pkl", 'rb') as f:
    dics = pickle.load(f)
    # i+1, 0 stand for padding elements
word_index_dic = {w: int(i) for w, i in dics.items()}
index_word_dic = {int(i): w for w, i in dics.items()}

with open("dataset/switchboard/tokenized.txt", 'r') as f:
    lines = f.readlines()[:100]
    for line in lines:
        labels_data = line.split()
        labels_data = Variable(torch.from_numpy(np.reshape([[word_index_dic.get(i, 1)] for i in labels_data], [len(labels_data),1])))
        labels_data = labels_data.cuda()
        #print labels_data.size() 
        
        outputs = s.decode(labels_data)       
        seq = ' '.join([index_word_dic[i] for i in outputs]) + '\n'
        print(seq)
               
 
