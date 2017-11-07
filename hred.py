from base import *

"""
embeddings: pretrained embedding tensor, vocab_size*embed_size
h_size: hidden size of encoder
c_size: hidden size of context RNN
d_size: hidden size of decoder
"""

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
class hred(base):
    def __init__(self, embeddings, h_size, c_size, d_size, lr=0.0002):
        super().__init__()
        #parameters
        self.vocab_size, self.embed_size = embeddings.size()
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=EOT)
        self.embedding.weight = nn.Parameter(embeddings)

        self.encoder = EncoderRNN(self.embedding, h_size, masked=True)
        self.crnn = ContextRNN(h_size, c_size)
        self.decoder = DecoderRNN(self.embedding, d_size, context_size=c_size, concated=True)
        #optimizer
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    #inputs: max_len*batch_size, Variable
    def forward(self, inputs):
        max_len, batch_size = inputs.size()
        e_hidden = Variable(torch.zeros(batch_size, self.encoder.hidden_size))
        c_hidden = Variable(torch.zeros(batch_size, self.crnn.hidden_size))
        d_hidden = Variable(torch.zeros(batch_size, self.decoder.hidden_size))
        
        c_inputs = torch.cat((EOT*Variable(torch.ones((1, batch_size)).long()), inputs[:-1, :]))
        current_mask = (c_inputs!=EOT).float()
        r_inputs = torch.cat((EOT*Variable(torch.ones((1, batch_size)).long()), c_inputs[:-1, :]))
        rolled_mask = (r_inputs!=EOT).float()
        outputs=[]
        for i in range(c_inputs.size(0)):
            r_m = rolled_mask[i]
            r_m.data.unsqueeze_(1)
            e_hidden = self.encoder(c_inputs[i], e_hidden, r_m)
            c_m = current_mask[i]
            c_m.data.unsqueeze_(1)
            c_hidden = self.crnn(e_hidden, c_hidden, c_m)
            d_hidden, o = self.decoder(c_inputs[i], d_hidden, c_hidden, c_m)
            outputs.append(o)
        return outputs#max_len*batch_size*vocab_size

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    em =torch.from_numpy(pickle.load(open(embed, 'rb'), encoding='latin1'))
    s=hred(em,h_size,c_size,d_size,lr)
    s=s.cuda()
    train_dialogs = pickle.load(open(train_pkl,'rb'))
    valid_dialogs = pickle.load(open(valid_pkl,'rb'))
    test_dialogs = pickle.load(open(test_pkl,'rb'))
    s.run_train(train_dialogs, valid_dialogs, num_epochs=50, b_size=b_size, check_dir = check_dir)
