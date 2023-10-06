from steps.build_dataset import NamesDataset
from steps.tokenizer import CharacterTokenizer

def test_dataset_build():
    words = ['adam','eve','gandhi','andrej']
    s = "".join(words)
    tokenizer = CharacterTokenizer(s)
    X,y = NamesDataset(context_length=3,tokenizer=tokenizer).get_dataset(words)
    assert X.shape[0] == y.shape[0]
