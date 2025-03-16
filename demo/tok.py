#import tiktoken
#encoding = tiktoken.get_encoding("cl100k_base")
#print(encoding.bos_token)
#encoded = encoding.encode("tiktoken is great!")
#print(encoded)

#from tokenizers import Tokenizer
#tokenizer = Tokenizer.from_pretrained("bert-base-cased")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#return_tensors="pt"
print(tokenizer.encode("tiktoken is great!",  return_tensors="pt"))
