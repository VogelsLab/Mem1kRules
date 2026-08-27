# Reproducible BND novelty/familiarity simulations

The `BND_IF_EEEIIEII_6pPol` pipeline compares responses to novel and familiar
input patterns. The corrected interface fixes three issues in the legacy
cluster workflow:

1. The Python/YAML option `N_input` now matches the C++ command-line option.
2. A BND rule is consistently treated as four rules with six coefficients
   each (24 values). Legacy 25-value arrays remain readable; the final
   background-rate nuisance value is ignored because BND fixes
   `rate_poisson` in its task configuration.
3. Randomness is controlled by one deterministic seed. The main simulator,
   `RandStimGroup`, and `RFConnection` no longer replace it with wall-clock
   seeds.

These changes can alter results relative to historical runs. That is expected:
the historical output ID did not control all random streams.

## Using the corrected branch

Clone the repository or update an existing clone:

```bash
git fetch origin
git switch --track origin/fix/bnd-reproducibility
```

If the branch already exists locally:

```bash
git switch fix/bnd-reproducibility
```

To return to the original legacy pipeline:

```bash
git switch main
```

After switching to the corrected branch, follow the build and usage instructions below.

## Build

Compile the simulator in the Auryn environment:

```bash
cd cluster_deployment/synapsbi/simulator/cpp_simulators
make -B sim_BND_IF_EEEIIEII_6pPol
```

Do not commit the compiled executable. Build it on the target workstation or
cluster so it links against the local Auryn, Boost, and MPI libraries.

## Command-line use

`--ID` names output files. `--seed` controls the stochastic realization:

```text
sim_BND_IF_EEEIIEII_6pPol \
  --ID rule_001 \
  --seed 202 \
  --N_input 10000 \
  ...
```

Use a unique output directory or ID for every run. If `--seed` is omitted, the
simulator deterministically derives a 32-bit fallback seed from `--ID`.

## Python use

The BND wrapper accepts 24-value and legacy 25-value theta vectors. Keep the
legacy `seeds` argument as the output identifiers and pass random seeds
separately:

```python
simulator.sample(
    thetas,
    seeds=rule_ids,
    simulation_seeds=[202] * len(thetas),
)
```

Parameter files support the same separation:

```python
make_param_files_cluster(
    simulator,
    thetas,
    rule_ids,
    "BND_IF_EEEIIEII_6pPol_params.txt",
    simulation_seeds=[202] * len(thetas),
)
```

## Verification performed on the ISTA cluster

The corrected sources were compiled against the project Auryn build and tested
with complete BND runs:

- three runs of one rule and seed produced byte-identical spike rasters and
  four final connection files;
- passing the fallback seed explicitly reproduced the fallback output;
- changing the explicit seed changed all six scientific output files;
- repeating the new seed reproduced those outputs exactly.

Run the lightweight wrapper regression tests with:

```bash
python -m unittest cluster_deployment/tests/test_bnd_reproducibility.py
```
