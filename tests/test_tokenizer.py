
from steps.tokenizer import CharacterTokenizer

def test_character_tokenizer_encode():
    s = "abcdefghijklmnopqrstuvwxyz"
    tokenizer = CharacterTokenizer(s)
    encode = tokenizer.encode("abcdefghijklmnopqrstuvwxyz")
    assert max(encode) == 26 and min(encode) == 1  

def test_character_tokenizer_decdoe():
    s = "abcdefghijklmnopqrstuvwxyz"
    l = range(1,27)
    tokenizer = CharacterTokenizer(s)
    decode = tokenizer.decode(l)
    assert max(decode) == 'z' 