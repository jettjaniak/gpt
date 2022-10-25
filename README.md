# GPT
Transformer Language Model; PyTorch implementation from scratch

It's intended to follow [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) architecture, i.e.:
 - decoder-only Transformer
 - learned positional encoding
 - dense multi-head attention
 - Normal(0, 0.02) initialization
 - d_mlp = 4 * d_head
 - embedding, residual and softmax dropouts

It's implemented as a collection of small modules, each depending on the one below it:
1. TransformerLM
2. Transformer
3. Decoder
4. MultiHeadAttention
5. Attention

They're all implemented in `gpt/`.  

Tensor shapes and types are annotated with [torchtyping](https://github.com/patrick-kidger/torchtyping), which is integrated with pytest and typeguard, so the types are automatically enforced in tests.

Each module except TransformerLM is tested. Tests usually just run forward / backward on a bunch of randomly shaped inputs, some of them check properties that should usually hold. Only attention has a typical unit test, actually comparing output values with ones I got on paper.  

I'm yet to train it on a toy dataset.
