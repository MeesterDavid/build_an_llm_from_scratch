from pathlib import Path
import re
from tokenizer import SimpleTokenizerV1
from importlib.metadata import version
import tiktoken
import torch


print("torch version: ", torch.__version__)
print(torch.cuda.is_available())


verdict_path = Path('data', 'the_verdict.txt')

print(f"path is {verdict_path}")

with open(verdict_path, "r") as verdict_file:
    verdict_text = verdict_file.read()

tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(verdict_text)
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]


# split = re.split(r'([,.:;?_!"()\']|--|\s)', verdict_text)
# split = [x for x in split if x.strip()]

# all_words   = sorted(set(split))
# all_words.extend(["<|endoftext|>", "<|unk|>"])
# vocab       = {token:integer for integer,token in enumerate(all_words)}

# tokenizer = SimpleTokenizerV1(vocab)
# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))
# print(text)

# ids = tokenizer.encode(text)
# print(ids)
# text = tokenizer.decode(ids)
# print(text)

# tokenizer = tiktoken.get_encoding("gpt2")
# text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the SUOMENADS."
# ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(ids)
# text = tokenizer.decode(ids)
# print(text)