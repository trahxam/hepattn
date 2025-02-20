# hep-attn

Goals:
- cleanly switch between different attention backends (sdpa, flash, flex)
- include some recent transformer advances (layerscale, value residuals, local attention)
- full torch.compile and nested tensor support
- don't use conda

## Setup

### Frist time

First install `pixi` according to https://pixi.sh/latest/.

This is probably just

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Clone the repo:

```bash
git clone git@github.com:samvanstroud/hepattn.git
```


To install, you need to first remove the `flash-attn` dependency from the `pyproject.toml` file.
Then run: 

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
pytest
```


## Run experiments

```bash
cd src/hepattn/experiments/trackml/
python hit_filter.py fit --config hit_filter.yaml --trainer.fast_dev_run 10
```

## Features

- [x] gated dense network
- [x] Flex transformer
- [x] Flex local
- [x] layerscale
- [x] value residuals including learnable per token
- [x] local flex with wrapping
- [x] hepformer positional embeddings
- [ ] fix flex with dynamic shapes
- [ ] mask decoder
- [ ] SAM random positional embeddings (possible to preseve symmetric posenc for phi)
- [ ] flex decoder
- [ ] flex mask attention
- [ ] flex local CA
- [ ] add pe to object queries and check impact on mask attention pattern
- [ ] alphafold2 attention gating
- [ ] register tokens but interspersed for local attention
- [ ] moe
- [ ] CLS token
- [ ] laser https://github.com/lucidrains/x-transformers/commit/57efd7770f2f5df0ff7b4ffcbd623750b584e850#diff-b335630551682c19a781afebcf4d07bf978fb1f8ac04c6bf87428ed5106870f5R2360
