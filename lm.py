from base import *

"""
embeddings: pretrained embedding tensor, vocab_size*embed_size
d_size: hidden size of decoder
independence: whether sentences are independent
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
check_dir = './lm'#check point file
######################
class lm(base):
    def __init__(self, embeddings, d_size, independence=False, lr=0.0002):
        super().__init__()
        #parameters
        self.vocab_size, self.embed_size = embeddings.size()
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=EOT)
        self.embedding.weight = nn.Parameter(embeddings)

        self.decoder = DecoderRNN(self.embedding, d_size, context_size=0,concated=False)
        
        self.independence = independence
        #optimizer
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    #inputs: max_len*batch_size, Variable
    def forward(self, inputs):
        max_len, batch_size = inputs.size()
        d_hidden = Variable(torch.zeros(self.batch_size, self.decoder.hidden_size))
        c_inputs = torch.cat((EOT*Variable(torch.ones((1, batch_size)).long()), inputs[:-1, :]))
        current_mask = (c_inputs!=EOT).float()
        outputs=[]
        for i in range(c_inputs.size(0)):
            c_m = current_mask[i]
            c_m.data.unsqueeze_(1)
            if self.independence:
                d_hidden, o = self.decoder(c_inputs[i], d_hidden, 0, c_m)
            else:
                d_hidden, o = self.decoder(c_inputs[i], d_hidden)
            outputs.append(o)
        return outputs#max_len*batch_size*vocab_size
    
    def decode(self, inputs, decode_length = 15):
        length = inputs.size()[0]
        d_hidden = Variable(torch.zeros(1, self.decoder.hidden_size))
        for i in range(length):
            d_hidden, o = self.decoder(inputs[i], d_hidden)
        decoder_outputs = []
        
        for i in range(decode_length):
            decoder_output = F.log_softmax(o)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOT: break
            decoder_outputs.append(ni)
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda()
            d_hidden, o = self.decoder(decoder_input, d_hidden)
        return decoder_outputs


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    em =torch.from_numpy(pickle.load(open(embed, 'rb'), encoding='latin1'))
    s=lm(em,d_size,independence=False, lr=lr)
    s=s.cuda()
    train_dialogs = pickle.load(open(train_pkl,'rb'))
    valid_dialogs = pickle.load(open(valid_pkl,'rb'))
    test_dialogs = pickle.load(open(test_pkl,'rb'))
    s.run_train(train_dialogs, valid_dialogs, num_epochs=50, b_size=b_size, check_dir = check_dir)
