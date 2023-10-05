import torch 
import pickle

from steps.utils import generate

config = {}
config.update({
    'context_window': 3,
})

if __name__ == "__main__": 

    #Load Model
    model = torch.load('saved_models/model.pth')

    #Loading Tokenizer
    fileObj = open('saved_models/tokenizer.obj', 'rb')
    tokenizer = pickle.load(fileObj)
    fileObj.close()

    # Getting Results
    for i in range(15):
        print(generate(tokenizer,model,config=config))