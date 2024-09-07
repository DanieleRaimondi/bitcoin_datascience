# Esegue i notebook uno dopo l'altro
jupyter nbconvert --to notebook --execute --inplace 1a.ThermoModel.ipynb
jupyter nbconvert --to notebook --execute --inplace 1b.LogTimeLogPrice.ipynb
jupyter nbconvert --to notebook --execute --inplace 1d.ThermoLogTimeLogPrice.ipynb
jupyter nbconvert --to notebook --execute --inplace 1e.SlopesGrowthModel.ipynb
jupyter nbconvert --to notebook --execute --inplace 2.Cycles.ipynb
jupyter nbconvert --to notebook --execute --inplace 3.DXY.ipynb
jupyter nbconvert --to notebook --execute --inplace 4.MVRV.ipynb
jupyter nbconvert --to notebook --execute --inplace 5.AvailableSupply.ipynb
jupyter nbconvert --to notebook --execute --inplace 6.Demand.ipynb
jupyter nbconvert --to notebook --execute --inplace 7.Cohorts.ipynb

echo "All the notebooks run."



# README
# 1) Entrare nel folder notebooks da terminale

# 2) Rendi lo Script Eseguibile (farlo solo la prima volta): 
#    chmod +x run_notebooks.sh

# 3) Lancia da terminale lo script per eseguire tutti i notebooks:
#    ./run_notebooks.sh



# Then, simply launch this:
# ~/bitcoin_datascience/notebooks/run_notebooks.sh