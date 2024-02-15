import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
import warnings
warnings.filterwarnings('ignore') # ou warnings.filterwarnings(action='once')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sklearn as sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn import metrics

st.set_page_config(layout = 'wide')

st.title("Modelo de Previsão de Vendas")

# Extração de dados ####################################################################################################################
lista_arquivos = list()
lista_arquivos.append('https://raw.githubusercontent.com/geo-vitoriano/sales_predict/main/dados/vendas_2020.zip')
lista_arquivos.append('https://raw.githubusercontent.com/geo-vitoriano/sales_predict/main/dados/vendas_2021.zip')
lista_arquivos.append('https://raw.githubusercontent.com/geo-vitoriano/sales_predict/main/dados/vendas_2022.zip')
lista_arquivos.append('https://raw.githubusercontent.com/geo-vitoriano/sales_predict/main/dados/vendas_2023.zip')

dataframes = []

for arquivo in lista_arquivos :
  df = pd.read_csv(arquivo, sep=';', compression='zip')
  dataframes.append(df)

dados = pd.concat(dataframes, ignore_index=True)
dados.sort_values(by=['Emissão Certificado'], inplace=True)

# Tratamento de dados #################################################################################################################

# Reduzindo o dataset para os campos que interessam na modelagem.
dados = dados[['Emissão Certificado', 'Desconto', 'Valor', 'Qtd', 'Atacado']]

# Renomeando os campos
dados = dados.rename(columns={"Emissão Certificado": "data",
                              "Valor": "valor",
                              "Qtd": "qtd",
                              "Atacado" : "atacado",
                              "Desconto": "desconto"})
# Alterando os tipos de dados
dados['data'] = pd.to_datetime(dados['data'], errors='coerce')
dados['valor'] = dados['valor'].str.replace(',','.').astype(float)
dados['desconto'] = dados['desconto'].astype(str)
dados['desconto'] = dados['desconto'].str.replace(',','.').astype(float)

# Alterando a informação do campo para binária.
for index, row in dados.iterrows() :
  if row['atacado'] == 'Sim' :
     dados.at[index, 'atacado'] = 1
  else :
    dados.at[index, 'atacado'] = 0

    # Gerando a totalização por dia e atacado.
grupo = dados.groupby(['data', 'atacado']).aggregate('sum')[['valor', 'desconto', 'qtd']]

l_data = list()
l_valor = list()
l_qtd = list()
l_desconto = list()
l_atacado = list()

for v in grupo.values :
  l_valor.append(v[0])
  l_desconto.append(v[1])
  l_qtd.append(v[2])

for v in grupo.index :
  l_data.append(v[0])
  l_atacado.append(v[1])

dados = pd.DataFrame({'data' :l_data,
                      'atacado' : l_atacado,
                      'valor' : l_valor,
                      'qtd' : l_qtd})

dados.sort_values(by=['data'], inplace=True)

# Criação de novos campos para a modelagem

dados = dados.reset_index(drop=True)

dados['dia_do_ano'] = dados['data'].dt.dayofyear
dados['mes'] = dados['data'].dt.month
dados['dia_semana'] = dados['data'].dt.dayofweek
dados['dia_do_mes'] = dados['data'].dt.day
dados['mes_ano'] = dados['data'].dt.month.map(str) + '/' + dados['data'].dt.year.map(str)
dados['ano'] = dados['data'].dt.year


# Retirando os outliars
filtro = dados['valor'] > 50.00
dados = dados[filtro]

# Construção do Modelo

def create_dataset(df):

    X = []
    y = []

    for i in range(len(df) - windows_size):

        pos_target = i + windows_size
        #print('pos_target ', pos_target )
        target = df.iloc[pos_target]['valor']

        sample = []
        for f in features:
            if f == 'valor' :
                sample += list(df.iloc[i:pos_target][f].values)
            else:
                sample += [df.iloc[pos_target][f]]

        X.append(sample)
        y.append(target)

    return np.array(X), np.array(y)

def get_model():
    #return LinearRegression()
    # return DecisionTreeRegressor(min_samples_leaf=15, max_depth=10)
    #return AdaBoostRegressor()
    # return RandomForestRegressor(n_jobs=-1, n_estimators=50, random_state=123, max_depth=8, max_features=0.15, min_samples_leaf=15)
     return RandomForestRegressor(n_jobs=-1, n_estimators=200, max_depth=10, max_features=0.15, min_samples_leaf=15)

def predict(sample, model):

    predictions = []
    # Valores das variáves dependentes
    test_info = X_test[:, windows_size:]

    sample_to_predict = [sample.copy()]

    for i in range(test_size):

        sample_to_predict = sample_to_predict[0]
        #print('sample_to_predict = ', sample_to_predict )

        if i > 0:
            # Valores dos target
            sales = list(sample_to_predict[1:windows_size]) + [y]
            # Valores das variáveis dependentes
            others = list(test_info[i])

            sample_to_predict = sales + others

        sample_to_predict = np.array([sample_to_predict])
        y = model.predict(sample_to_predict)[0]

        predictions.append(y)

    return predictions

# Parâmetros do modelo  ###############################################################################################################################

windows_size = 365 # ref. day
val_size = 0 # ref. day
test_size = 30 # ref. day
train_size = len(dados) - val_size - test_size

features = [ 'valor',	'qtd', 'atacado', 'mes', 'dia_do_ano', 'dia_semana', 'dia_do_mes', 'ano']
y_real = list(dados['valor'].iloc[-test_size:].values)

#Criando o conjunto de treino e teste  ################################################################################################################
X, y = create_dataset(dados)
X_test, y_test = X[-test_size:, :], y[-test_size:]
X_val, y_val = X[-test_size-val_size:-test_size, :], y[-test_size-val_size:-test_size]
X_train, y_train = X[-test_size-val_size-train_size:-test_size-val_size, :], y[-test_size-val_size-train_size:-test_size-val_size]

X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape

# Gerando a previsão  #################################################################################################################################

# Instância do modelo
model = get_model()
# Ajuste do modelo na correlação
model.fit(X_train, y_train)
# Previsão do modelo
#y_pred = model.predict(X_test)
y_pred = predict(X_test[0,:], model)

# Métricas de Erros ###################################################################################################################################

mae = mean_absolute_error(y_pred, y_test)
mse = mean_squared_error(y_pred, y_test)

# Gráfico comparando valores reais com previstos ######################################################################################################

dias = range(0, 30)

df_pred = pd.DataFrame(data = { 'dias' : dias, 'previsto': y_pred })
df_pred = df_pred.reset_index(drop = True)


df_real = pd.DataFrame(data = { 'dias' : dias, 'real': y_real })
df_real = df_real.reset_index(drop = True)


df_final = pd.merge( df_pred, df_real, how='inner', on='dias' )

fig = px.line(df_final, x=df_final.index, y=df_final.columns[1:], markers = True)
fig.update_legends()
fig.update_layout(
    xaxis_title='Dias',
    yaxis_title='R$ Valor',
    legend=dict( orientation="v")
)


st.plotly_chart(fig, use_container_width = True)