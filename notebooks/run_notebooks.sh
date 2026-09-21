#!/usr/bin/env bash
#
# Re-runs every notebook in place, refreshing the charts under ../output.
#
# Usage, from this directory:
#   chmod +x run_notebooks.sh   # first time only
#   ./run_notebooks.sh

set -euo pipefail

cd "$(dirname "$0")"

# Prefer the project's virtualenv when it exists.
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
run_nb 1i.ThermoModel_v2.ipynb
run_nb 2a.Cycles.ipynb
run_nb 2b.CyclesNorm.ipynb
run_nb 2c.MVRV.ipynb
run_nb 3a.Economics.ipynb
run_nb 3b.DXY.ipynb
run_nb 4a.Supply.ipynb
run_nb 4c.Demand.ipynb
#run_nb 5.Cohorts.ipynb
run_nb 6a.GoogleTrends.ipynb
run_nb 7a.BTCvsUSElections.ipynb
run_nb 7b.US_Elections.ipynb
run_nb 8.ETF_Inflows.ipynb
run_nb 9.BTC_Miners.ipynb
run_nb 10a.PuellMultiple.ipynb
run_nb 10b.MVRVZScoreNUPL.ipynb
run_nb 10c.PiCycleTop.ipynb
run_nb 10d.GoldenRatioMultiplier.ipynb
run_nb 10e.HashRibbons.ipynb
run_nb 10f.2YearMAMultiplier.ipynb
run_nb 11.RiskModel.ipynb
# run_nb 99.DynamicPlot.ipynb

echo "All notebooks executed."
