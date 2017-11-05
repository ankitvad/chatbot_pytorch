from base import *

"""
embeddings: pretrained embedding tensor, vocab_size*embed_size
d_size: hidden size of decoder
independence: whether sentences are independent
"""

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
        d_hidden = Variable(torch.zeros(batch_size, self.decoder.hidden_size))
        c_inputs = torch.cat((EOT*Variable(torch.ones((1, batch_size)).long()), inputs[:-1, :]))
        current_mask = (c_inputs!=EOT).float()
        outputs=[]
        for i in range(c_inputs.size(0)):
            c_m = current_mask[i]
            c_m.data.unsqueeze_(1)
            if self.independence:
                d_hidden, o = self.decoder(inputs[i], d_hidden, 0, c_m)
            else:
                d_hidden, o = self.decoder(inputs[i], d_hidden)
            outputs.append(o)
        return outputs#max_len*batch_size*vocab_size
    
if __name__ == '__main__':
    em=torch.ones((9,3))
    s=lm(em,3,3)
    dialogs = [[1,2,3,4,0,5,6,7] for i in range(20)]
    print(dialogs)
    trained = dialogdata(dialogs)
    validated = dialogdata(dialogs)
    for epoch in range(1,3):
        train_dataloader = DataLoader(trained, batch_size=6, shuffle=True)
        valid_dataloader = DataLoader(trained, batch_size=6, shuffle=True)
        s.train(train_dataloader, epoch)
        s.validate(valid_dataloader)
