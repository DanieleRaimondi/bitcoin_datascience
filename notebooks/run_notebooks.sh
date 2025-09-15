# Esegue i notebook uno dopo l'altro
jupyter nbconvert --to notebook --execute --inplace 1a.ThermoModel.ipynb
jupyter nbconvert --to notebook --execute --inplace 1b.LogTimeLogPrice.ipynb
jupyter nbconvert --to notebook --execute --inplace 1d.EnsembleCorridor.ipynb
jupyter nbconvert --to notebook --execute --inplace 1e.SlopesGrowthModel.ipynb
jupyter nbconvert --to notebook --execute --inplace 1g.Growths.ipynb
jupyter nbconvert --to notebook --execute --inplace 1h.Metcalfe.ipynb
jupyter nbconvert --to notebook --execute --inplace 2a.Cycles.ipynb
jupyter nbconvert --to notebook --execute --inplace 2b.CyclesNorm.ipynb
jupyter nbconvert --to notebook --execute --inplace 2c.MVRV.ipynb
jupyter nbconvert --to notebook --execute --inplace 3a.Economics.ipynb
jupyter nbconvert --to notebook --execute --inplace 3b.DXY.ipynb
jupyter nbconvert --to notebook --execute --inplace 4a.Supply.ipynb
jupyter nbconvert --to notebook --execute --inplace 4c.Demand.ipynb
jupyter nbconvert --to notebook --execute --inplace 5.Cohorts.ipynb
jupyter nbconvert --to notebook --execute --inplace 6a.GoogleTrends.ipynb
jupyter nbconvert --to notebook --execute --inplace 7a.BTCvsUSELECTIONS.ipynb
jupyter nbconvert --to notebook --execute --inplace 7b.US_Elections.ipynb
jupyter nbconvert --to notebook --execute --inplace 8.ETF_Inflows.ipynb
jupyter nbconvert --to notebook --execute --inplace 9.BTC_Miners.ipynb
#jupyter nbconvert --to notebook --execute --inplace 99.DynamicPlot.ipynb

echo "All the notebooks run."



# README
# 1) Entrare nel folder notebooks da terminale

# 2) Rendi lo Script Eseguibile (farlo solo la prima volta): 
#    chmod +x run_notebooks.sh

# 3) Lancia da terminale lo script per eseguire tutti i notebooks:
#    ./run_notebooks.sh



# Then, simply launch this:
# ~/bitcoin_datascience/notebooks/run_notebooks.sh