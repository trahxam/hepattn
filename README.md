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
- [x] local flex with wrapping
- [ ] flex decoder
- [ ] flex mask attention
- [ ] flex local CA
- [ ] hepformer positional embeddings
- [ ] SAM random positional embeddings
- [ ] add pe to object queries and check impact on mask attention pattern
- [ ] alphafold2 attention gating
- [ ] register tokens but interspersed for local attention
- [ ] moe
- [ ] CLS token

## Notes

- einops doesn't work with nested tensors
    - support masking?
    - don't use einops?
