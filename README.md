# CHA-CF: Class Hierarchy Adaptation via Confusion Feedback

This repository provides a self-contained MATLAB reproduction of the
**AWAphog × HFS-MIMR** setting reported for CHA-CF. The retained pipeline is:

```text
AWAphog → MIMR feature selection → supplied hierarchy H0 →
bottom-up local hierarchy adaptation H1
```

The release deliberately contains one runnable setting. It does not include
other base HFS methods, datasets, exploratory variants, or historical results.

## Requirements

- MATLAB R2025b on Apple Silicon macOS
- Statistics and Machine Learning Toolbox
- Bioinformatics Toolbox
- The included Apple Silicon LIBSVM MEX binaries in `private/`

For Windows, Linux, or Intel Mac, replace `private/svmtrain.mexmaca64` and
`private/svmpredict.mexmaca64` with compatible LIBSVM MATLAB binaries.

## Quick start

In MATLAB, add only this repository's root directory to the path:

```matlab
addpath('/path/to/CHA-CF');
result = run_awa_mimr();
```

The runner resolves the dataset paths relative to its own location, so the
current MATLAB working directory does not matter. A typical run takes about
three minutes.

To reproduce and verify the published values at four decimal places, run:

```matlab
ok = check_run_awa_mimr();
```

The check returns `true` only if all six metrics and the accepted-block count
match the expected values.

## Expected results

| Path | Acc | F_H | TIE | FLCA |
| --- | ---: | ---: | ---: | ---: |
| Baseline hierarchy | 0.2449 | 0.5737 | 0.3410 | 0.4958 |
| CHA-CF | 0.2664 | 0.7327 | 0.1552 | 0.6247 |

CHA-CF accepts two local blocks for this setting.

## Reproduction configuration

| Parameter | Value |
| --- | ---: |
| Holdout ratio (`rho`) | 0.25 |
| Fusion coefficient (`lambda`) | 0.30 |
| Random seed | 13 |
| Candidate-pair shortlist (`K_pair`) | 8 |
| Development evaluations (`K_dev`) | 5 |
| MIMR parameters (`lambda`, `alpha`, `beta`, `maxIte`) | 10, 0.1, 0.01, 10 |
| Classifier | Linear SVM (`-c 1 -t 0 -q`) |
| Maximum global passes | 2 |
| Maximum accepted blocks per layer | 1 |

## Repository layout

- `run_awa_mimr.m`: main entry point.
- `check_run_awa_mimr.m`: reproduction check.
- `run_relift_case.m` and `relift_*.m`: CHA-CF implementation.
- `private/`: tree utilities, hierarchical metrics, MIMR helpers, and LIBSVM binaries.
- `datasets/`: AWAphog training and test files required by the runner.

## Scope

This code is intended to reproduce the released AWAphog × HFS-MIMR experiment.
Please cite the accompanying manuscript when using the implementation or data
preparation in academic work.
