#Importamos Dependencias o Bibliotecas necesarias:
import pyreadstat # librería para leer formato ".dta"
import pandas as pd # librería para manipulación de datos
import numpy as np # Librería para operaciones matemáticas
import matplotlib # Librería para graficar
from matplotlib import pyplot as plt # Librería para graficar
import statsmodels.api as sm # Librería para análisis estadístico
from IPython.display import Image # Librería para importar imagénes
from statsmodels.formula.api import ols # Para pruebas de hipotesis

dtafile = 'Data/cgreene76.dta'
# metodo para leer
dataframe , meta = pyreadstat.read_dta(dtafile)
print(dataframe.head(5))
dataframe['Ltotcost'] = np.log(dataframe['costs'])
dataframe['Loutput'] = np.log(dataframe['output'])
dataframe["Loutput_2"] = dataframe["Loutput"]**2
dataframe['Lplabor'] = np.log(dataframe['plabor'])
dataframe['Lpfuel'] = np.log(dataframe['pfuel'])
dataframe['Lpkap'] = np.log(dataframe['pkap'])
dataframe['One'] = 1
print(dataframe.head(10))
Y = dataframe["Ltotcost"]
# que valores me faltan
X = dataframe[["One","Loutput","Loutput_2", "Lplabor", "Lpfuel", "Lpkap"]]
est = sm.OLS(Y,X)
est2 = est.fit()
print(est2.summary())

print("\n Parametros \n ",est2.params)
print("\n Valores \n", est2.tvalues)
print("\n Fvalue \n", est2.fvalue)

R = np.array(([0,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]))
print(est2.f_test(R))

formula = 'Ltotcost ~  One + Loutput + Loutput_2 + Lplabor + Lpfuel + Lpkap '
#results = sm.OLS(formula, dataframe)
print("\n\n\n")
#formula = 'Ltotcost ~  One + Loutput + Lplabor + Lpfuel + Lpkap'
results = ols(formula, dataframe).fit()

hypotheses = 'Lplabor + Lpfuel + Lpkap = 1'

t_test = results.t_test(hypotheses)
print("\n****************** Hipotesis\n\n*************")
print(t_test)

# Otro ejemplo: 
hypotheses_2 = 'Lplabor + Lpfuel + Lpkap = 1, Loutput = 0'
f_test = results.f_test(hypotheses_2)
print(f_test)



dataframe.to_csv('guardar.csv')