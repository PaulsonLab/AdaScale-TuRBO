# AdaScale-TuRBO

Code for the paper [Rethinking Trust Region Bayesian Optimization in High Dimensions](https://arxiv.org/abs/2604.22967),
accepted to the 2026 AISTATS OPTIMAL Workshop.

## Installation

```sh
pip install -r requirements.txt
```

## Running experiments

Experiments are launched with `main.py`. A benchmark must be specified:

```sh
python main.py benchmark=<benchmark_name>
```

Available benchmarks (see [`configs/benchmark/`](configs/benchmark)):

| Benchmark     | Dim | Domain           | Budget |
|---------------|-----|------------------|--------|
| `rastrigin`   | 50  | `[-5, 5]^d`      | 500    |
| `michalewicz` | 50  | `[0, 3.14]^d`    | 500    |
| `schwefel`    | 50  | `[-500, 500]^d`  | 500    |
| `rover`       | 60  | `[-0.1, 1.1]^d`  | 1000   |

Adding `seed=<number>` is recommended for reproducibility.

### Configuration overrides

Default settings live in [`configs/default.yaml`](configs/default.yaml) and the per-benchmark files.
Since this project uses [Hydra](https://hydra.cc/), any value can be overridden from the command line:

```sh
# Override the evaluation budget for the Rastrigin benchmark
python main.py benchmark=rastrigin seed=0 benchmark.n_tot=1000
```

Hydra's multirun mode sweeps over seeds or benchmarks:

```sh
python main.py -m benchmark=rastrigin,schwefel seed=0,1,2
```

Run outputs and logs are written to `outputs/<timestamp>/`.

## Project structure

```
main.py                          # Hydra entry point
configs/
  default.yaml                   # Global settings (seed, Hydra output dirs)
  benchmark/*.yaml               # Per-benchmark settings
src/
  AdaScale_TuRBO_optimize.py     # Trust-region state, candidate generation, BO loop
  gpytorch_modules_new.py        # Kernel with TR- and dimension-scaled lengthscale prior
  benchmark/                     # Schwefel and rover test functions
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tang2026rethinking,
  title={Rethinking Trust Region Bayesian Optimization in High Dimensions},
  author={Tang, Wei-Ting and Paulson, Joel A},
  journal={arXiv preprint arXiv:2604.22967},
  year={2026}
}
```

## License

MIT; see [LICENSE](LICENSE).
