from abc import ABC,abstractmethod
import torch

class Dataset(ABC):

    @abstractmethod
    def get_dataset(self):
        pass


class NamesDataset(Dataset):

    def __init__(self,context_length,tokenizer):
        self.context_length = context_length
        self.tokenizer = tokenizer

    def get_dataset(self,words):
        X = []
        Y = []

        for w in words:
            new_w = "".join(["."]*self.context_length) + w + '.'
            encode_w = self.tokenizer.encode(new_w)
            for i in range(len(w)+1):
                X.append(encode_w[i:i+self.context_length])
                Y.append(encode_w[i+self.context_length])

        return torch.tensor(X),torch.tensor(Y)



