# Esegue i notebook uno dopo l'altro
jupyter nbconvert --to notebook --execute --inplace 1.ThermoModel.ipynb
jupyter nbconvert --to notebook --execute --inplace 2.Cycles.ipynb
jupyter nbconvert --to notebook --execute --inplace 3.LogTimeLogPrice.ipynb
jupyter nbconvert --to notebook --execute --inplace 4.DXY.ipynb
jupyter nbconvert --to notebook --execute --inplace 5.MVRV.ipynb
jupyter nbconvert --to notebook --execute --inplace 6.AvailableSupply.ipynb
jupyter nbconvert --to notebook --execute --inplace 7.Demand.ipynb
jupyter nbconvert --to notebook --execute --inplace 8.Cohorts.ipynb

echo "All the notebooks run."



# README
# 1) Entrare nel folder notebooks da terminale

# 2) Rendi lo Script Eseguibile (farlo solo la prima volta): 
#    chmod +x run_notebooks.sh

# 3) Lancia da terminale lo script per eseguire tutti i notebooks:
#    ./run_notebooks.sh



# Then, simply launch this:
# ~/bitcoin_datascience/notebooks/run_notebooks.sh