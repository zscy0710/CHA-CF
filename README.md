# CHA-CF

MATLAB code for reproducing the AWAphog experiment in *Class Hierarchy
Adaptation via Confusion Feedback*.

The included experiment uses HFS-MIMR, the supplied class hierarchy, and
bottom-up CHA-CF adaptation.

## Requirements

- MATLAB R2025b on Apple Silicon macOS
- Statistics and Machine Learning Toolbox
- Bioinformatics Toolbox
- The included Apple Silicon LIBSVM MEX files in `private/`

For Windows, Linux, or Intel Mac, replace `private/svmtrain.mexmaca64` and
`private/svmpredict.mexmaca64` with compatible LIBSVM MATLAB binaries.

## Run

Add the repository root to the MATLAB path and run:

```matlab
addpath('/path/to/CHA-CF');
result = run_awa_mimr();
```

The runner locates the dataset files relative to its own directory. A run
takes several minutes on the environment above.

To check the reported values, run:

```matlab
ok = check_run_awa_mimr();
```

## Expected results

| Method | Acc | F_H | TIE | FLCA |
| --- | ---: | ---: | ---: | ---: |
| HFS-MIMR | 0.2449 | 0.5737 | 0.3410 | 0.4958 |
| HFS-MIMR + CHA-CF | 0.2664 | 0.7327 | 0.1552 | 0.6247 |

CHA-CF accepts two local hierarchy updates in this run.

## Configuration

| Parameter | Value |
| --- | ---: |
| Holdout ratio | 0.25 |
| Fusion coefficient | 0.30 |
| Random seed | 13 |
| Candidate-pair shortlist | 8 |
| Development evaluations | 5 |
| MIMR parameters (lambda, alpha, beta, maxIte) | 10, 0.1, 0.01, 10 |
| Classifier | Linear SVM (`-c 1 -t 0 -q`) |
| Maximum global passes | 2 |
| Maximum accepted updates per layer | 1 |

## Files

- `run_awa_mimr.m`: experiment entry point.
- `check_run_awa_mimr.m`: result check.
- `run_relift_case.m` and `relift_*.m`: CHA-CF implementation.
- `private/`: HFS-MIMR utilities, hierarchical metrics, and LIBSVM binaries.
- `datasets/`: AWAphog training and test data.
