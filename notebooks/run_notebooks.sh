#!/usr/bin/env bash

set -euo pipefail

# Esegue i notebook uno dopo l'altro usando la .venv del progetto se disponibile.
if [ -x "../.venv/bin/python" ]; then
	PY_CMD="../.venv/bin/python"
else
	PY_CMD="python"
fi

run_nb() {
	"$PY_CMD" -m jupyter nbconvert --to notebook --execute --inplace "$1"
}

run_nb 1a.ThermoModel.ipynb
run_nb 1b.LogTimeLogPrice.ipynb
run_nb 1d.EnsembleCorridor.ipynb
run_nb 1e.SlopesGrowthModel.ipynb
run_nb 1g.Growths.ipynb
run_nb 1h.Metcalfe.ipynb
run_nb 2a.Cycles.ipynb
run_nb 2b.CyclesNorm.ipynb
run_nb 2c.MVRV.ipynb
run_nb 3a.Economics.ipynb
run_nb 3b.DXY.ipynb
run_nb 4a.Supply.ipynb
run_nb 4c.Demand.ipynb
#run_nb 5.Cohorts.ipynb
run_nb 6a.GoogleTrends.ipynb
run_nb 7a.BTCvsUSELECTIONS.ipynb
run_nb 7b.US_Elections.ipynb
run_nb 8.ETF_Inflows.ipynb
run_nb 9.BTC_Miners.ipynb
# run_nb 99.DynamicPlot.ipynb

echo "All notebooks executed."



# README
# 1) Entrare nel folder notebooks da terminale

# 2) Rendi lo Script Eseguibile (farlo solo la prima volta): 
#    chmod +x run_notebooks.sh

# 3) Lancia da terminale lo script per eseguire tutti i notebooks:
#    ./run_notebooks.sh



# Then, simply launch this:
# ~/bitcoin_datascience/notebooks/run_notebooks.sh