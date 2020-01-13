[![codecov](https://codecov.io/gh/CINPLA/visual-stimulation/branch/dev/graph/badge.svg)](https://codecov.io/gh/CINPLA/visual-stimulation)
[![Build Status](https://travis-ci.org/CINPLA/visual-stimulation.svg?branch=dev)](https://travis-ci.org/CINPLA/visual-stimulation)


# visual-stimulation
Analysis for experiments with visual stimulation e.g. drifting gratings.


### Dependencies

- `python >=3.5`
- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `pytest`
- `elephant`
- `neo`
- `quantities`
- `exdir`
- `expipe` (needed in [getting_started_tutorial.ipynb](https://github.com/CINPLA/visual-stimulation/blob/dev/examples/getting_started_tutorial.ipynb))

### Examples
- A getting started tutorial can be found here: [getting_started_tutorial.ipynb](https://github.com/CINPLA/visual-stimulation/blob/dev/examples/getting_started_tutorial.ipynb)

- [Other examples](https://github.com/CINPLA/visual-stimulation/blob/dev/examples/examples.ipynb)


### Developers
- `analysis.py`: contains different analysis functions (e.g. `compute_dsi`)
- `utils.py`: contains functions for organizing ànd getting data (e.g. create trials)
- `helper.py`: contains helper functions.
- `axona_helper.py`: extracts stimulus data from axona specific files. A similar function to `generate_stim_group_and_epoch()` should be implemented if other setups are used.
