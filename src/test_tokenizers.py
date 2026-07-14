import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import re
    from tokenizer import SimpleTokenizerV1
    from data_loader import create_dataloader_v1, tokenizer
    import torch
    from importlib.metadata import version
    import os
    print(os.getcwd())
    print("torch version: ", torch.__version__)
    print(torch.cuda.is_available())
    return Path, create_dataloader_v1, tokenizer, torch


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(Path):
    notebook_dir = Path(__file__).parent.parent
    notebook_dir
    verdict_path = Path(notebook_dir, 'data', 'the_verdict.txt')
    print(f"path is {verdict_path}")
    with open(verdict_path, "r") as verdict_file:
        verdict_text = verdict_file.read()
    return (verdict_text,)


@app.cell
def _(create_dataloader_v1, tokenizer, torch, verdict_text):
    vocab_size = tokenizer.n_vocab
    output_dim = 256
    max_length = 4
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    dataloader = create_dataloader_v1(
        verdict_text,
        batch_size=8,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )
    data_iterator = iter(dataloader)
    inputs, targets = next(data_iterator)
    token_embeddings = embedding_layer(inputs)
    return max_length, output_dim, token_embeddings


@app.cell
def _(max_length, output_dim, torch):
    context_length = max_length
    positional_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    positional_embeddings = positional_embedding_layer(torch.arange(context_length))
    return (positional_embeddings,)


@app.cell
def _(positional_embeddings, token_embeddings):
    input_embeddings = token_embeddings + positional_embeddings
    input_embeddings.shape
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
