# Esegue i notebook uno dopo l'altro
jupyter nbconvert --to notebook --execute --inplace 1.ThermoModel.ipynb
jupyter nbconvert --to notebook --execute --inplace 2.Cycles.ipynb
jupyter nbconvert --to notebook --execute --inplace 3.MVRV.ipynb
jupyter nbconvert --to notebook --execute --inplace 4.Cohorts.ipynb
jupyter nbconvert --to notebook --execute --inplace 5.AvailableSupply.ipynb

echo "Tutti i notebook sono stati eseguiti."



# README
# 1) Entrare nel folder notebooks da terminale

# 2) Rendi lo Script Eseguibile (farlo solo la prima volta): 
#    chmod +x run_notebooks.sh

# 3) Lancia da terminale lo script per eseguire tutti i notebooks:
#    ./run_notebooks.sh



#POI OGNI VOLTA LANCIARE PIU' COMODAMENTE SOLO QUESTO:
# ~/bitcoin_datascience/notebooks/run_notebooks.sh