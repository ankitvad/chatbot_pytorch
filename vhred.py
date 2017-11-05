from base import *

"""
embeddings: pretrained embedding tensor, vocab_size*embed_size
h_size: hidden size of encoder
c_size: hidden size of context RNN
d_size: hidden size of decoder
l_size: size of latent variable
"""

class vhred(base):
    def __init__(self, embeddings, h_size, c_size, d_size, l_size, lr=0.0002):
        super().__init__()
        #parameters
        self.vocab_size = embeddings.size(0)
        self.embed_size = embeddings.size(1)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=EOT)
        self.embedding.weight = nn.Parameter(embeddings)

        self.l_size = l_size
        self.encoder = EncoderRNN(self.embedding, h_size, masked=True)
        self.crnn = ContextRNN(h_size, c_size)
        self.decoder = DecoderRNN(self.embedding, d_size, context_size=c_size+l_size, concated=True)
        
        self.prior = GaussPara(l_size, c_size)
        self.post = GaussPara(l_size, c_size+h_size)
        #optimizer
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    #inputs: max_len*batch_size, Variable
    def forward(self, inputs):
        max_len = inputs.size(0)
        batch_size = inputs.size(1)
        e_hidden = Variable(torch.zeros(batch_size, self.encoder.hidden_size))
        c_hidden = Variable(torch.zeros(batch_size, self.crnn.hidden_size))
        d_hidden = Variable(torch.zeros(batch_size, self.decoder.hidden_size))
        c_inputs = torch.cat((EOT*Variable(torch.ones((1, batch_size)).long()), inputs[:-1, :]))
        current_mask = (c_inputs!=EOT).float()
        r_inputs = torch.cat((EOT*Variable(torch.ones((1, batch_size)).long()), c_inputs[:-1, :]))
        rolled_mask = (r_inputs!=EOT).float()

        hs, cs = [], []
        outputs=[]
        kl = Variable(torch.zeros((max_len, batch_size)))
        for i in range(c_inputs.size(0)):
            r_m = rolled_mask[i]
            r_m.data.unsqueeze_(1)
            hs.append(self.encoder(c_inputs[i], e_hidden, r_m))
            e_hidden = self.encoder(c_inputs[i], e_hidden, r_m)
            c_m = current_mask[i]
            c_m.data.unsqueeze_(1)
            cs.append(self.crnn(e_hidden, c_hidden, c_m))
            c_hidden = self.crnn(e_hidden, c_hidden, c_m)

        r_hs = self.mask_state(hs, inputs)#max_len*batch_size*h_size
        for i in range(c_inputs.size(0)):
            c_h = torch.cat((cs[i], r_hs[i]), 1)
            post_mu, post_cov = self.post(c_h)
            post_z = self.sample(post_mu, post_cov)

            pri_mu, pri_cov = self.prior(cs[i])
            c_kl = self.kl(post_mu, pri_mu, post_cov, pri_cov)
            kl[i] = c_kl*(1-current_mask[i])
            
            d_hidden, o = self.decoder(inputs[i], d_hidden, torch.cat((cs[i], post_z), 1), c_m)
            outputs.append(o)
        return outputs, kl#outputs: max_len*batch_size*vocab_size; kl: max_len*batch_size

    #targets: max_len*batch_size
    #outputs: max_len*batch_size*vocab_size
    def cost(self, targets, outputs, kl, loss=nn.CrossEntropyLoss()):
        t_loss=0
        mask = (targets>0).float()
        t_len=torch.sum(mask)
        t_kl = torch.sum(kl)
        r_input = torch.cat((EOT*Variable(torch.ones((1, targets.size(1))).long()), targets[:-1, :]))
        num_eot = torch.sum((r_input==EOT).float())
        for i in range(targets.size(0)):
            t_loss+=torch.sum(loss(outputs[i], targets[i]))
        return t_loss, t_loss/t_len, t_kl, t_kl/num_eot

if __name__ == '__main__':
    em=torch.ones((6,3))
    s=vhred(em,3,3,3,1)
    inputs=Variable(torch.ones((10,10)).long())
    for batch in range(10):
        o, kl = s(inputs)
        c=s.cost(inputs, o, kl)
        print(c[0])
        print(c[2])
        s.train(c[0]+c[2])
