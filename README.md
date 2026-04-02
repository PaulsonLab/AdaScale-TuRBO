# AdaScale-TuRBO
This is the repo containing codes for the "Rethinking Trust Region Bayesian Optimization in High Dimensions" paper.
This work has been accepted to 2026 AISTATS OPTIMAL Workshop.

# Installation
```sh
pip install -r requirements.txt
```

## Running Experiments

Experiments can be run using the `main.py` script. You must specify a benchmark to run the algorithms.

**Basic Command**
```
python main.py benchmark=<benchmark_name>
```

*   To see a list of available benchmarks, run `python main_NeSTBO.py`.
*   Adding `seed=<number>` is recommended for reproducibility.

**Configuration Overrides**

All default settings are stored in configs/default.yaml. Since this project uses [Hydra](https://hydra.cc/), you have the flexibility to modify these values on the fly via the command line without editing the file.

```
# Example: override the evaluation budget for the rastrigin benchmark
python main.py benchmark=rastrigin seed=0 benchmark.n_tot=1000
```


