# Dynamic Time Warping como metodologia para treinamento de modelos de previsão para Temperatura da Superfície Oceânica
## Trabalho Prático da Matéria SCC5977 - Aprendizado de Máquina para Séries Temporais
### Integrantes
- [@maiserjose](https://github.com/maiserjose)
- [@matheusvictal](https://github.com/matheusvictal)
- [@vitorfrois](https://github.com/vitorfrois)

## Resumo do Projeto
Neste trabalho estudamos os modelos DS (DTW [[2]](#2) + SVR [[1]](#2)), SARIMA  e LSTM em relação aos erros quadráticos médios e os erros absolutos percentual médios (MAPE) para cada conjunto de treino e teste definidos na realização das previsões das temperaturas.

Os dados de interesse no projeto se tratam de séries temporais de temperatura da superfície ocânica disponibilizadas pela NOAA (National Oceanic and Atmospheric Administration). Mais especificamente, utilizamos os [dados da base de reconstrução de temperatura da superfície oceânica](https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html) disponibilizada para a comunidade específica. 

## Repositório
Os arquivos finais do projeto estão no diretório `/source` enquanto `/legado` contém os arquivos de projeto usados durante o desenvolvimento do projeto final. 

Para reproduzir os códigos localmente crie um ambiente virtual e instale as bibliotecas utilizadas
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Referências
<a id="1">[1]</a> 
Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E. (2011). 
Scikit-learn: Machine Learning in Python. 
Journal of Machine Learning Research, 12, 2825-2830.

<a id="2">[2]</a> 
T. Giorgino. Computing and Visualizing Dynamic Time Warping Alignments in R: The dtw Package.
  J. Stat. Soft., doi:10.18637/jss.v031.i07.








