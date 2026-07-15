# Hodgkin-Huxley Parameter Estimation

Estimates the leak conductance ($g_L$) and leak reversal potential ($E_L$) of a neuron from an experimental voltage trace, while $g_K$/$E_K$ (and $g_{Na}$/$E_{Na}$) are held at standard squid-axon values.

Two ways to use it:

- **`app.py`** — interactive Streamlit web app (upload a file, tweak parameters, see plots live)
- **`hh.py`** — command-line/batch runner that saves plots and CSVs to disk

Both sit on top of **`hh_core.py`**, which holds all the model math (gating kinetics, windowed regression, Monte-Carlo sampling, parametric KL divergence) and has no plotting or printing side effects.

## Method

1. **Dynamic (windowed) estimation** — a sliding-window linear regression isolates the leak current after subtracting the modeled Na⁺/K⁺ currents, recovering $g_L$ and $E_L$ over time.
2. **Voltage distribution fitting** — the experimental voltage trace is compared against histogram, KDE, and best-fit-Gaussian representations.
3. **Monte-Carlo objective sampling** — random $(g_L, E_L)$ pairs are simulated and scored by RMSE and by the KL divergence between parametric fits of the experimental and simulated voltage distributions.

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

(Optional but recommended: do this inside a virtual environment — `python -m venv venv` then `source venv/bin/activate` on Mac/Linux or `venv\Scripts\activate` on Windows.)

## Usage

### Web app

```bash
streamlit run app.py
```

This opens a browser tab. Upload a voltage data file (`.xlsx` or `.csv`), adjust the parameters in the sidebar/columns, and click **Run Analysis**.

### Command line

Open `hh.py` and edit the `USER CONFIG` block near the top (file path, column number, time step, window/step size, etc.), then run:

```bash
python hh.py
```

Plots and CSVs are written to `./hh_outputs`.

## Data format

Voltage data files should have paired time/voltage columns. `column_number` is 1-indexed and maps to the underlying column index as `2 * column_number - 2`, matching the source data's paired layout. Excel files are read with the first 3 rows skipped as headers.

## Project structure

```
.
├── app.py         # Streamlit GUI
├── hh.py          # CLI batch runner
├── hh_core.py     # Estimation engine (model math, no I/O side effects)
├── requirements.txt
└── README.md
```

## Requirements

See `requirements.txt`. Core dependencies: Streamlit, NumPy, pandas, SciPy, scikit-learn, Matplotlib, openpyxl (for reading `.xlsx` files).
