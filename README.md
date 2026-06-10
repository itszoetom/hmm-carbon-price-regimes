# Hidden Markov Regime-Switching Models for EU Carbon Allowance Prices

A hidden Markov model of EU ETS carbon allowance (EUA) daily log-returns, with
latent market regimes (for example normal, stressed, crashed). The estimated
transition matrix is then analyzed with finite-state Markov chain theory to find
the stationary distribution and the most likely hidden regime path.

## Repository layout

    params.py            all tunable parameters in one place
    data/
      processed/         cleaned series with daily log-returns
    scripts/
      data_loader.py     single entry point for reading and preparing data
      eda/               exploratory plots and summary statistics
      counting/          rule-based regimes and counted transition matrix
      hmm/               Baum-Welch fitting and model selection by BIC
      markov/            stationary distribution, durations, convergence, sensitivity
      decode/            Viterbi decoding
      plotting/          shared figure helpers
    figures/             generated plots
    tables/              generated tables
    docs/                project proposal and final paper

## Setup

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Getting the data

The data for this project is from the EUA front December futures daily settlement, downloaded from
Investing.com's [Carbon Emissions Futures](https://www.investing.com/commodities/carbon-emissions-historical-data)
page. Download the daily series into data/raw and prepare it with:

    python -m scripts.data_loader

This writes the cleaned series to data/processed and prints a data-quality
summary.

## Reproducing the analysis

After preparing the data, run the scripts as modules from the repository root,
in order:

    python -m scripts.eda.run_eda
    python -m scripts.counting.run_counting
    python -m scripts.hmm.run_hmm
    python -m scripts.markov.run_markov
    python -m scripts.decode.run_decode

Each run script reads the processed data, calls the functions in its method
folder, and writes its figures to figures/ and tables to tables/. The hmm script
must run before the markov and decode scripts, since they use its saved model and
transition matrix.