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
        self.tanh = nn.Tanh()
        #parameters
        self.vocab_size = embeddings.size(0)
        self.embed_size = embeddings.size(1)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=EOT)
        self.embedding.weight = nn.Parameter(embeddings)

        self.l_size = l_size
        self.encoder = EncoderRNN(self.embedding, h_size, masked=True)
        self.crnn = ContextRNN(h_size, c_size)
        self.decoder = DecoderRNN(self.embedding, d_size, context_size=c_size+h_size, concated=True)
        
        self.prior = GaussPara(l_size, c_size)
        self.post = GaussPara(l_size, c_size+h_size)

        self.reconst = nn.Linear(l_size, h_size)
        #optimizer
        self.embed_optim = torch.optim.Adam(self.embedding.parameters(), lr=lr)
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.crnn_optim = torch.optim.Adam(self.crnn.parameters(), lr=lr)
        self.decoder_optim = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        self.prior_optim = torch.optim.Adam(self.prior.parameters(), lr=lr)
        self.post_optim = torch.optim.Adam(self.post.parameters(), lr=lr)
        self.reconst_optim = torch.optim.Adam(self.reconst.parameters(), lr=lr)

        self.wake_optimizer = [self.embed_optim, self.encoder_optim, self.crnn_optim, self.decoder_optim, self.prior_optim]
        self.sleep_optimizer = [self.prior_optim, self.post_optim, self.reconst_optim]

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
        r_loss = Variable(torch.zeros((max_len, batch_size)))
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
            reconstructed = self.tanh(self.reconst(post_z))
            r_loss[i] = torch.sum((reconstructed - r_hs[i])**2, 1)*(1-current_mask[i])*0.5
            
            c_m = current_mask[i]
            c_m.data.unsqueeze_(1)
            
            d_hidden, o = self.decoder(inputs[i], d_hidden, torch.cat((cs[i], reconstructed), 1), c_m)
            outputs.append(o)
        return outputs, kl, r_loss#outputs: max_len*batch_size*vocab_size; kl&r_loss: max_len*batch_size

    #targets: max_len*batch_size
    #outputs: max_len*batch_size*vocab_size
    def cost(self, targets, outputs, kl, rloss, loss=nn.CrossEntropyLoss()):
        t_loss=0
        mask = (targets>0).float()
        t_len=torch.sum(mask)
        t_kl = torch.sum(kl)
        t_rloss = torch.sum(rloss)
        r_input = torch.cat((EOT*Variable(torch.ones((1, targets.size(1))).long()), targets[:-1, :]))
        num_eot = torch.sum((r_input==EOT).float())
        for i in range(targets.size(0)):
            t_loss+=torch.sum(loss(outputs[i], targets[i]))
        return t_loss, t_loss/t_len, t_kl, t_kl/num_eot, t_rloss, t_rloss/num_eot

    #state: wake or sleep
    def optimize(self, loss, state):
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)
        if state=='wake':
            for optim in self.wake_optimizer:
                optim.zero_grad()
                optim.step()
        else:
            for optim in self.sleep_optimizer:
                optim.zero_grad()
                optim.step()

if __name__ == '__main__':
    em=torch.ones((6,3))
    s=vhred(em,3,3,3,1)
    inputs=Variable(torch.ones((10,10)).long())
    for batch in range(10):
        o, kl, rloss = s(inputs)
        c=s.cost(inputs, o, kl, rloss)
        print(c[0])
        print(c[2])
        print(c[4])
        if batch%2==0:
            s.optimize(c[0]+c[2]+c[4], 'wake')
        else:
            s.optimize(c[0]+c[2]+c[4], 'sleep')
