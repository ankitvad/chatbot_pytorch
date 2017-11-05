from seq2seq import *
import numpy as np
from dataload import *

if __name__ == '__main__':
    em=torch.ones((9,3))
    s=seq2seq(em,3,3)
    dialogs = [[1,2,3,4,0,5,6,7] for i in range(20)]
    print(dialogs)
    trained = dialogdata(dialogs)
    validated = dialogdata(dialogs)

    for epoch in range(20):
        #train
        dataloader = DataLoader(trained, batch_size=6, shuffle=True)
        for i, batch in enumerate(dataloader):
            inputs = Variable(batch.t())
            print(inputs.size())
            c=s.cost(inputs, s(inputs))
            print(c[0])
            s.train(c[0])
        #validate
        dataloader = DataLoader(validated, batch_size=6, shuffle=False)
        for i, batch in enumerate(dataloader):
            inputs = Variable(batch.t())
            print(inputs.size())
            c=s.cost(inputs, s(inputs))
            print(c[0])
