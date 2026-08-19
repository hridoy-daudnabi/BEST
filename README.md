# Mode Choice Share Prediction Tool

An open, transparent toolkit for estimating a multinomial logit (MNL) mode choice
model and exploring how changes in level-of-service attributes shift the aggregate
mode share. It ships with a synthetic 1,000-traveler dataset, a dependency-light MNL
estimator, and a Streamlit app with sliders that recompute predicted shares live.

Seven modes are considered: Drive, Carpool, TNC (Uber/Lyft), Public Transit, Bike,
Walk, and Park & Ride.

## What it does

- Generates a synthetic choice dataset from a documented data generating process.
- Estimates a simple MNL by maximum likelihood, respecting alternative availability.
- Reports predicted mode shares and the implied value of time.
- Lets you scale travel time, per-trip cost, and monthly parking cost for any mode
  by a percentage and see how overall shares respond, both in aggregate and for a
  single traveler.

## Modeling approach

Utility for person *i* choosing alternative *j*:

```
V_ij = ASC_j + b_time * time_ij + b_cost * cost_ij + b_park * park_ij
```

- Cost enters every mode except Bike and Walk, which are free.
- Monthly parking cost enters only Drive, Carpool, and Park & Ride, each with its own
  coefficient so it can be varied directly in the tool.
- Bike and Walk utility is driven by distance through travel time; they carry no cost
  or parking term.
- Walk is available only when trip distance is under 3 miles, Bike only under 6 miles.
  Availability is enforced in both estimation and prediction.

The synthetic data are drawn from known coefficients (see `modechoice/config.py`),
so estimation can be checked against ground truth. On the shipped sample the estimator
recovers the level-of-service coefficients close to their true values and reproduces
the observed mode-share marginals. The rarer alternatives carry noisier constants, as
expected from their smaller counts.

## Install

```bash
git clone <your-repo-url>
cd mode-choice-tool
pip install -e .
```

Requires Python 3.9+. Dependencies: numpy, pandas, scipy, streamlit, altair.

## Usage

Regenerate the dataset:

```bash
python -m modechoice.generate_data
```

Estimate the model and print a fit summary:

```bash
python -m modechoice.mnl
```

Launch the interactive tool:

```bash
streamlit run app/streamlit_app.py
```

The app has three tabs: **Policy sensitivity** (sliders plus a base-vs-scenario share
chart), **Model** (estimated coefficients, t-statistics, value of time, observed vs
predicted shares), and **Individual traveler** (per-person choice probabilities under
the current sliders).

### Static web tool (HTML)

A self-contained browser version lives in `docs/`. It shows the attribute controls,
the predicted mode-share table, and the base-vs-scenario chart, with no server to run.
Rebuild its data whenever the model changes:

```bash
python -m modechoice.export_web      # writes docs/model_data.js from the fitted model
```

Then open `docs/index.html` in a browser, or host it for free on GitHub Pages:
in the repo settings under Pages, set the source to the `main` branch and the `/docs`
folder. The estimation runs in Python; the page applies the fitted coefficients and
recomputes shares client-side, so it stays exact and fast.

Use it as a library:

```python
import pandas as pd
from modechoice import estimate, predicted_shares, sensitivity

df = pd.read_csv("data/synthetic_mode_choice.csv")
result = estimate(df)
print(predicted_shares(df, result.params))

# Effect of a 25% increase in Drive parking on aggregate shares
print(sensitivity(df, result.params, "Drive", "park", 0.25))
```

## Repository layout

```
modechoice/            core package
  config.py            modes, availability rules, true coefficients
  generate_data.py     synthetic data generator
  mnl.py               estimation, prediction, sensitivity
  export_web.py        exports the fitted model to docs/model_data.js
app/
  streamlit_app.py     interactive tool (Python)
docs/
  index.html           static web tool (HTML/CSS/JS)
  model_data.js        generated: fitted coefficients + attribute matrices
tests/
  test_modechoice.py   generation, estimation, and sensitivity tests
data/
  synthetic_mode_choice.csv
```

## Data dictionary

One row per traveler. For each mode `M` the columns are `time_M` (minutes),
`cost_M` (dollars per trip), `park_M` (dollars per month), and `av_M` (1 if the mode
is available to that person). Plus `person_id`, `distance_mi`, and `chosen_mode`.

## Tests

```bash
pytest
```

## License

MIT. See `LICENSE`.
