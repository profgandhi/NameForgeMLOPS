import torch
from torch.nn import functional as F

@torch.no_grad()
def generate(tokenizer,model,config):
    '''
    Function to generate names from model
    '''
    out = []
    context = tokenizer.encode(".") * config['context_window']
    while True:
        #forward pass
        logits = model(torch.tensor(context))
        prob = F.softmax(logits,dim=1)

        #Sampling from the distribution with replacement
        idx = torch.multinomial(prob, 1, replacement=True).item()
        context = context[1:] + [idx]
        out.append(idx)

        if(idx == tokenizer.encode(["."])[0]):
            break

    return "".join(tokenizer.itos[i] for i in out)