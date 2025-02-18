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

## Notes

- einops doesn't work with nested tensors
    - support masking?
    - don't use einops?

- thinking about diagnalising the maskattention operator
    - note mask attention is not attention. it's just the dot product of vectors
    - i can apply my own "score mod" to that if I want to penalise off diagonal terms
    - but how is this approach different from using LCA? basically how is alibi different from LCA?
    - and by extension, how would doing manual PE be different from using LCA?
    - TODO: go back to LCA test in hepformer but compare to no maskattention, not to maskattention
        - look into some basic metrics like how many hits are selected by MA etc
        - test if perf is impacted by LCA window size much
    - TODO: try with just the standard sinusoidal PE or random PE (don't worry about wrapping the PE) and see if it works
            (might be enough to use the PE and then apply wrapping with the mask)
    - if that doesn't work -- just switch to a flex mask attention implementation