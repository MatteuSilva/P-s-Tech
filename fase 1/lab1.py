import pandas as pd

leitura = pd.read_csv('sih_cnv_qiuf112852177_221_129_62.csv', encoding='ISO-8859-1', skiprows=3, sep=';', thousands='.', decimal=',', skipfooter=12, usecols=['2024', 'Total'])

print(leitura)