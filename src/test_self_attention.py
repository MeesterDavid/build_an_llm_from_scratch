import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import torch

    return (torch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Your journey starts with one step embedded in a tensor
    """)
    return


@app.cell
def _(torch):
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]]
    )
    return (inputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Variable Definitions

    * **$X$** : The input sequence matrix, where the sequence tokens are represented as:
      $$x^{(1)}, x^{(2)}, \dots, x^{(T)}$$

    * **$W$** : The attention score matrix, where an individual element $w_{i,t}$ represents the attention score (or weight) of input token $t$ with respect to query token $i$:
      $$w_{i,t} = \text{Attention}(q_i, k_t)$$

      We determine the attention scores by computing the dot product of the query $x^{(2)}$ with every other input token.
    """)
    return


@app.cell
def _(inputs, torch):
    query = inputs[1]
    attention_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attention_scores_2[i] = torch.dot(x_i, query)
    attention_scores_2
    return (attention_scores_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Normalized
    """)
    return


@app.cell
def _():
    # attention_weights_2 = attention_scores_2 / attention_scores_2.sum(0)
    # attention_weights_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Softmaxed
    softmax normaliseert, maar zorgt ook dat alle weights positief zijn. Bovendien is het geschikter voor gradient descent vanwege de mooie afgeleide $p *(1 - p)$
    """)
    return


@app.cell
def _(attention_scores_2, torch):
    def softmax_naive(x):
        return torch.exp(x)/torch.exp(x).sum(dim=0)

    attention_weights_2_naive = softmax_naive(attention_scores_2)
    attention_weights_2_naive
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using Pytorch
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get the context vector for the second token by multiplying the embedded input token $x^{(1)}$ - which we call the query - with the corresponding attention weights $w_{1,0}, \dots , w_{1,T}$
    When getting the total context vector for a sequence we multiply all embedded input tokens and their corresponding weights and sum the resulting vectors.
    """)
    return


@app.cell
def _(attention_scores_2, inputs, torch):
    attention_weights_2 = torch.softmax(attention_scores_2, dim=0)
    attention_weights_2
    query_2 = inputs[1]
    context_vector_2 = torch.zeros(query_2.shape)

    for j in range(len(inputs)):
        context_vector_2 += attention_weights_2[j] * inputs[j]

    print(context_vector_2)

    context_vector_2 = (attention_weights_2 @ inputs).sum()
    
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
