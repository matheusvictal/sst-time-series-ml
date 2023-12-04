import torch
import numpy as np
import torch.nn as nn
import math
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def mape(y_true, y_pred) -> float:
        return mean_absolute_percentage_error(y_true, y_pred)

def default_plot(y_real, y_pred, point_name: str, ml_model_name: str):
        """
        point_name: [Indian Ocean, Atlantic Ocean, etc]
        ml_model_name: [SVR, SARIMA, LSTM]
        """
        # Imprime as previsoes para o conjunto de teste e os valores reais
        fig, ax = plt.subplots(figsize=(15, 2.5))
        sns.set(style="whitegrid")

        color_dict = {
            'SVR': 'deeppink',
            'SARIMA': 'chartreuse',
            'LSTM': 'orangered'
        }

        try:
            color_to_use = color_dict[ml_model_name]
        except KeyError:
            color_to_use = 'gray'
        
        x_index = [
            '2022-01',
            '2022-02',
            '2022-03',
            '2022-04',
            '2022-05',
            '2022-06',
            '2022-07',
            '2022-08',
            '2022-09',
            '2022-10',
            '2022-11',
            '2022-12'
        ]

        df_lines = pd.DataFrame(
            {'Data': x_index, 
            'Actual SST': y_real,
            'Predicted SST': y_pred})

        df_lines.set_index('Data', inplace=True, drop=True)

        sns.lineplot(data=df_lines,
                    palette={'Actual SST': 'indigo', 'Predicted SST': color_to_use},
                    linewidth=1.5)

        plt.xticks(rotation=45)
        plt.title(f'SST prediction for {point_name}', fontsize=16)
        plt.xlabel('Time indicator (test set)', fontsize=12)
        plt.ylabel('SST', fontsize=12)
        plt.show()

def create_split_data(ts, lookback, test_start):
    data_raw = ts.to_numpy() # convercao da série para numpy
    data = []
    # laço para criar todas as combinacoes possiveis de comprimento igual a 'lookback'
    for index in range(len(data_raw) - lookback):
        # cada observaca 't' ira conter o pontos anteriores de 'lookbak'
        data.append(data_raw[index: index + lookback])

    data = np.array(data)

    # vetores para redes recorrentes possuem tres dimensoes:
    # - numero de amostras (para treino e teste)
    # - numero de passos no tempo (deinido pelo parametro 'lookback')
    # - numero de variaveis (1 para o caso de serie univariada)

    # o x e o y irao ter um deslocamento entre si de forma que cada ponto de
    # 0 ate t-1 sera utilizado para prever o ponto t
    x_train = data[:test_start, :-1, :]
    y_train = data[:test_start, -1, :]

    x_test = data[test_start:, :-1, :]
    y_test = data[test_start:, -1, :]

    # conversao para tensores pytorch
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    return [x_train, y_train, x_test, y_test]



class LSTM_base(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim):
        super(LSTM_base, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(in_features=hidden_size, out_features=output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])

        return out
    

def training_and_evaluate(model, epochs, loss_function, optimizer, X_train, y_train, X_test, verbose=True):
    # A variavel history_train e utilizada para armazenar a perda em cada epoca
    history_train = np.zeros(epochs)

    # colocar no modelo no tipo treino
    model.train()

    # o modelo sera treinado utilizando um batch, portanto, ha apenas um 
    # laço para iterar as epocas no procedimento de teste 
    for epoch in range(epochs):

        # forward pass
        y_pred = model.forward(X_train)

        # funcao perda
        loss = loss_function(y_pred, y_train)

        # o modo verboso imprime a informacao de treinamento
        if verbose:
            print(f"Epoch [{epoch + 1}|{epochs}] Loss: {loss.item()}")

        # inclui a perda da epoca atual no history_train
        history_train[epoch] = loss.item()

        # limpa o gradiente
        optimizer.zero_grad()

        # atualiza os parametros
        loss.backward()
        optimizer.step()

    # coloca o modelo em modo de validacao
    model.eval()

    # desativa o caluclo do gradiente. isso e util para procedimentos inferenciais.
    # doc.: (https://pytorch.org/docs/stable/generated/torch.no_grad.html)
    with torch.no_grad():
        # realiza predicoes para os conjuntos de treino e de teste
        y_pred_train = model.forward(X_train)
        y_pred_test = model.forward(X_test)

    return history_train, y_pred_train, y_pred_test


def model_performance(scaler, y_train, y_hat_train, y_test, y_hat_test, point_name, model_name):
    
    # aplica a transformacao inversa nos dados normalizados
    y_train_rev = scaler.inverse_transform(y_train.detach().numpy()).tolist()
    y_hat_train_rev = scaler.inverse_transform(y_hat_train.detach().numpy()).tolist()
    y_test_rev = scaler.inverse_transform(y_test.detach().numpy()).tolist()
    y_hat_test_rev = scaler.inverse_transform(y_hat_test.detach().numpy()).tolist()

    # calcula e obtem o RMSE
    train_RMSE = math.sqrt(mean_squared_error(y_train_rev, y_hat_train_rev))
    test_RMSE = math.sqrt(mean_squared_error(y_test_rev, y_hat_test_rev))

    print('Train score: {result} RMSE'.format(result=train_RMSE))
    print('Test score: {result} RMSE'.format(result=test_RMSE))

    # calcula e obtem o MAPE
    train_MAPE = mape(y_train_rev, y_hat_train_rev)
    test_MAPE = mape(y_test_rev, y_hat_test_rev)

    print('Train score: {result} MAPE'.format(result=train_MAPE))
    print('Test score: {result} MAPE'.format(result=test_MAPE))

    default_plot(y_real=[value for sublist in y_test_rev for value in sublist], 
                 y_pred=[value for sublist in y_hat_test_rev for value in sublist],
                 point_name=point_name,
                 ml_model_name=model_name)
