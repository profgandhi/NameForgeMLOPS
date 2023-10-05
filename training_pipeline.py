from steps.data_ingestion import IngestData
from steps.tokenizer import CharacterTokenizer
from steps.build_dataset import NamesDataset
from steps.model import MLP
from steps.utils import generate

import torch
import pickle

config = {}
config.update({
    'context_window': 3,
    'emb_size' : 2,
    'd_model' : 100,
    'epochs' : 10000
})

if __name__ == "__main__": 
    names = IngestData(data_path="names.csv").get_data()
    tokenizer = CharacterTokenizer(" ".join(names))
    config.update({
        'vocab_size' : len(tokenizer.vocab),
    })
    X,y = NamesDataset(context_length=config['context_window'],tokenizer=tokenizer).get_dataset(names)
    model = MLP(config=config)
    model,pl = model.train(X,y)

    for i in range(15):
        print(generate(tokenizer,model,config=config))

    #Saving tokenizer
    fileObj = open('saved_models/tokenizer.obj', 'wb')
    pickle.dump(tokenizer,fileObj)
    fileObj.close()

    #Saving model
    torch.save(model, 'saved_models/model.pth')

     
