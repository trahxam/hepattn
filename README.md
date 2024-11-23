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

test



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
