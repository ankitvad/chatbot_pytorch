from base import *

"""
embeddings: pretrained embedding tensor, vocab_size*embed_size
h_size: hidden size of encoder
c_size: hidden size of context RNN
d_size: hidden size of decoder
"""

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
        for i in range(r_inputs.size(0)):
            r_m = rolled_mask[i]
            r_m.data.unsqueeze_(1)
            e_hidden = self.encoder(c_inputs[i], e_hidden, r_m)
            c_m = current_mask[i]
            c_m.data.unsqueeze_(1)
            c_hidden = self.crnn(e_hidden, c_hidden, c_m)
            d_hidden, o = self.decoder(inputs[i], d_hidden, c_hidden, c_m)
            outputs.append(o)
        return outputs#max_len*batch_size*vocab_size


if __name__ == '__main__':
    em=torch.ones((6,3))
    s=hred(em,3,3,3)
    inputs=Variable(torch.ones((10,10)).long())
    for batch in range(10):
        c=s.cost(inputs, s(inputs))
        print(c[0])
        s.train(c[0])
