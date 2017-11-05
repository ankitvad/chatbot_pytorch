import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import inspect

USE_CUDA = True

EOT=2


"""
masked: whether mask hidden states over sequences, default true
embeddings: vocab_size*embed_size, type nn.Embedding
"""
class EncoderRNN(nn.Module):
    def __init__(self, embeddings, hidden_size, masked=True):
        super().__init__()
        #parameters
        self.embedding = embeddings
        self.hidden_size = hidden_size
        self.masked = masked
        self.gru = nn.GRUCell(embeddings.weight.size(1), hidden_size)

    """
    inputs: batch_size
    initial hidden: 0
    mask: EOT state of last word
    """
    def forward(self, inputs, hidden, mask=1):
        embedded = self.embedding(inputs)
        if self.masked:
            hidden = mask*hidden
        return self.gru(embedded, hidden)#batch_size*hidden_size


class ContextRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        #parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(input_size, hidden_size)

    """
    inputs: batch_size*input_size
    initial hidden: 0
    mask: EOT state of current word
    """
    def forward(self, inputs, hidden, mask):
        new_h = self.gru(inputs, hidden)
        return new_h*(1-mask)+hidden*mask#batch_size*hidden_size

"""
input_size: size of context except word embedding
embeddings: vocab_size*embed_size, type nn.Embedding
contexted: whether exist context
concated: whether concat context vector to each step
"""
class DecoderRNN(nn.Module):
    def __init__(self, embeddings, hidden_size, context_size=0, concated=True):
        super().__init__()
        #parameters
        self.vocab_size, self.embed_size = embeddings.weight.size()
        self.embedding = embeddings
        self.input_size = context_size+self.embed_size if concated else self.embed_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.concated = concated
        self.gru = nn.GRUCell(self.input_size, hidden_size)
        self.decode1 = nn.Linear(hidden_size, self.embed_size)
        self.decode2 = nn.Linear(self.embed_size, self.vocab_size)
        if self.context_size>0:
            self.initS = nn.Linear(context_size, hidden_size)
        self.tanh = nn.Tanh()

    """
    input_word: batch_size
    context: batch_size*context_size
    initial hidden: 0
    mask: EOT state of current word
    """
    def forward(self, input_word, hidden, context=0, mask=1):
        hidden = self.tanh(self.initS(context))*(1-mask)+hidden*mask if self.context_size>0 else hidden*mask
        if self.concated:
            inputs = torch.cat((context, self.embedding(input_word)), 1)
        else:
            inputs = self.embedding(input_word)
        new_h = self.gru(inputs, hidden)#batch_size*hidden_size
        output1 = self.tanh(self.decode1(new_h))
        output2 = self.tanh(self.decode2(output1))#batch_size*vocab_size
        return new_h, output2

"""
lsize: size of latent variable
csize: size of context vector
learn mean and covariance of Gaussian latent variable
"""
class GaussPara(nn.Module):
    def __init__(self, lsize, csize):
        super().__init__()
        #parameters
        self.firstl = nn.Linear(csize, lsize)
        self.secondl = nn.Linear(lsize, lsize)
        self.mu = nn.Linear(lsize, lsize)
        self.cov = nn.Linear(lsize, lsize)
        self.nonlinear = nn.Tanh()
        self.softplus = nn.Softplus()
        self.scale_cov = 0.01

    #context: batch_size*csize
    def forward(self, context):
        v = self.nonlinear(self.secondl(self.nonlinear(self.firstl(context))))
        mu = self.mu(v)
        cov = self.softplus(self.cov(v))*self.scale_cov
        return mu, cov
        
class base(nn.Module):
    def __init__(self):
        super().__init__()
        print('WTF')
        print(inspect.getargspec(nn.CrossEntropyLoss))
    
    #mask utterance as the state of the cloest forward one
    def mask_state(self, states, inputs):
        max_len = len(states)
        outputs = [0]*max_len
        new_h = states[-1]
        for i in range(max_len-1, -1, -1):
            mask = (inputs[i]==EOT).float()#batch_size
            mask.data.unsqueeze_(1)
            new_h = states[i]*mask + new_h*(1-mask)
            outputs[i] = new_h
        return outputs

    #targets: max_len*batch_size
    def sample(self, mu, cov):
        e = Variable(torch.randn(mu.size()))
        return mu+e*torch.sqrt(cov)

    #kl(p1||p2)
    def kl(self, mu1, mu2, s1, s2):
        kl = 0.5 * (torch.sum(torch.log(torch.abs(s2)) - torch.log(torch.abs(s1)) + s1 / s2 + (mu2 - mu1) ** 2 / s2, 1) - mu1.size(1))
        return kl
    
    #outputs: max_len*batch_size*vocab_size
    def cost(self, targets, outputs, loss=nn.CrossEntropyLoss()):
        t_loss=0
        mask = (targets>0).float()
        t_len=torch.sum(mask)
        for i in range(targets.size(0)):
            t_loss+=torch.sum(loss(outputs[i], targets[i]))
        return t_loss, t_loss/t_len

    def train(self, loss):
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optim.step()