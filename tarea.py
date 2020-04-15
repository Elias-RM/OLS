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
dataframe['Lpprod'] = np.dot(dataframe['plabor'] , dataframe['pfuel'])
dataframe['Lptcostfuel'] = np.dot(dataframe['Loutput'] , dataframe['pfuel'])
dataframe['avgcost'] = dataframe["costs"]/dataframe["output"]
dataframe['One'] = 1
print(dataframe.head(10))

Y = dataframe["Ltotcost"]
# que valores me faltan
#X = [["One","Loutput","Loutput_2","Lplabor","Lpfuel", "Lpkap","Lpprod","Lptcostfuel"]]
X = dataframe[["One","Loutput","Loutput_2", "Lplabor", "Lpfuel", "Lpkap","Lpprod","Lptcostfuel"]]

#X = [["One","Loutput", "Lplabor", "Lpfuel", "Lpkap"]]


est = sm.OLS(Y,X)
est2 = est.fit()
print(est2.summary())
print("\n\n\n=====================\n   OLS\n=====================\n\n\n")
print("\n Parametros \n ",est2.params)
print("\n Valores \n", est2.tvalues)
print("\n Fvalue \n", est2.fvalue)

R = np.array(([0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1] ))
print(est2.f_test(R))

formula = 'Ltotcost ~  One + Loutput  + Lplabor + Lpfuel + Lpkap + Lpprod + Lptcostfuel'
#results = sm.OLS(formula, dataframe)

#formula = 'Ltotcost ~  One + Loutput + Lplabor + Lpfuel + Lpkap'
results = ols(formula, dataframe).fit()

print("\n\n\n=====================\n            Hipotesis 1\n=====================\n\n\n")
hypotheses = 'Lplabor + Lpfuel + Lpkap = 1'
t_test = results.t_test(hypotheses)
print(t_test)
f_test = results.f_test(hypotheses)
print(f_test)

# Otro ejemplo: 
print("\n\n\n=====================\n            Hipotesis 2\n=====================\n\n\n")
hypotheses_2 = 'Lplabor + Lpfuel + Lpkap = 1, Lptcostfuel = 0'
t_test = results.t_test(hypotheses_2)
print(t_test)
f_test = results.f_test(hypotheses_2)
print(f_test)

# Otro ejemplo: 
print("\n\n\n=====================\n            Hipotesis 3\n=====================\n\n\n")
hypotheses_2 = 'Lplabor + Lpfuel + Lpkap = 1, Lpprod = 0'
hypotheses_2 = 'Lplabor + Lpfuel + Lpkap = 1, Lptcostfuel = 0'
t_test = results.t_test(hypotheses_2)
print(t_test)
f_test = results.f_test(hypotheses_2)
print(f_test)

print("\n==== Graficas")
LY_pred = est2.predict(X)
# Anti-log:
Y = np.exp(LY_pred)
# Colocamos en el Data Frame:
dataframe['totcost_e'] = Y
dataframe['avgcost_e'] = dataframe["totcost_e"]/dataframe["output"]
dataframe.head()
# graficamos resultados:
dataframe ['costs'].plot.kde()
plt.title("Densidad totcost MM USD")
plt.show()

plt.scatter(dataframe.output, dataframe.avgcost, s = 15, color ="red")
plt.scatter(dataframe.output, dataframe.avgcost_e, s = 15, color ="blue")
plt.title("Gráfico de dispersión Output vs Avg cost / Avg cost estimado")
#
plt.show()

dataframe.to_csv('guardar.csv')