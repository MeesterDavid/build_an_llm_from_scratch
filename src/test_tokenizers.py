import marimo

app = marimo.App()


@app.cell
def __():
    from pathlib import Path
    import re
    from tokenizer import SimpleTokenizerV1
    from data_loader import create_dataloader_v1, tokenizer
    import torch
    from importlib.metadata import version
    return (
        Path,
        SimpleTokenizerV1,
        create_dataloader_v1,
        re,
        tokenizer,
        torch,
        version,
    )


@app.cell
def __(torch):
    print("torch version: ", torch.__version__)
    print(torch.cuda.is_available())
    return


@app.cell
def __(Path):
    verdict_path = Path('data', 'the_verdict.txt')
    print(f"path is {verdict_path}")
    return (verdict_path,)


@app.cell
def __(verdict_path):
    with open(verdict_path, "r") as verdict_file:
        verdict_text = verdict_file.read()
    return (verdict_text,)


@app.cell
def __(tokenizer, torch):
    vocab_size = tokenizer.n_vocab
    output_dim = 256
    max_length = 4
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    return embedding_layer, max_length, output_dim, vocab_size


@app.cell
def __(create_dataloader_v1, max_length, verdict_text):
    dataloader = create_dataloader_v1(
        verdict_text,
        batch_size=8,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )
    return (dataloader,)


@app.cell
def __(dataloader):
    data_iterator = iter(dataloader)
    inputs, targets = next(data_iterator)
    print("Token IDs: \n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    return data_iterator, inputs, targets


if __name__ == "__main__":
    app.run()

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