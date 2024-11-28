# hep-attn

## Setup

### Frist time

First install `pixi` according to https://pixi.sh/latest/.

This is probably just

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Clone the repo:

```bash

```


To install, you need to first remove the flash attention dependency from teh `pyproject.toml` file. Then run: 

```bash
pixi install
```

Then, add back the flash attention dependency and install again.

### Coming back
On DIAS: 

```bash
cd /home/xucapsva/hepattn
pixi shell
python src/hepattn/flex.py
exit
```

## Run tests

```bash
pytest tests/
```



## Features

- [x] gated dense network
- [x] Flex transformer
- [x] Flex local
- [x] layerscale
- [x] value residuals including learnable per token
- [ ] local flex with wrapping
- [ ] flex decoder
- [ ] Flex mask attention
- [ ] Flex local CA
- [ ] SAM random positional embeddings
- [ ] alphafold2 attention gating
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