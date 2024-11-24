# hep-attn

## Setup

First install `pixi` according to https://pixi.sh/latest/.

On DIAS: 

```bash
cd /home/xucapsva/hepattn
pixi shell
python src/hepattn/flex.py
exit
```

Check tests

```bash
pytest tests/
```



## Features

- [x] gated dense network
- [x] Flex transformer
- [x] Flex local
- [x] layerscale
- [ ] flex decoder
- [ ] Flex mask attention
- [ ] Flex local CA
- [ ] SAM random positional embeddings
- [ ] alphafold2 attention gating
- [ ] value residuals including learnable per token
- [ ] register tokens but interspersed for local attention
- [ ] moe
- [ ] CLS token

## Notes

- einops doesn't work with nested tensors
    - support masking?
    - don't use einops?
- Need to compile block mask creation to avoid materialising the entire mask. howevber block mask is recompiled when sequence length changes. so can't really use this for the hit filter.
    - probably okay to run on the filtered MxN with with the full mask materialised, but now this means only gains are in time complexity (assuming we can diagonalise the mask attention). Space will grow as NxM. 
    - use FA2 for local attn
    - how does it handle NJT??
    - need to ask in an issue for details and future plans