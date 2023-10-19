import numpy as np
import pandas as pd
import utils
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_covtype
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_lfw_people
from tensorflow.keras.datasets import cifar10

#### DESCRIÇÃO
### O trabalho em questão, em suma, encontra duas envoltórias convexas em um dataset e testa sua sobreposição.
### Se positivo, printa no terminal que elas estão sobrepostas. Se negativo, então tentamos encontrar uma reta
### entre seus pontos mais próximos. Note que pode não ser possível traçar uma reta, no caso de uma reta completamente
### vertical (pois não há função que mapeia y para mais de um x) ou completamente horizontal (pois ocorreria uma divisão
### por 0 ao tentar calcular o coeficiente angular). Caso seja possível, então plotamos as envoltórias e as retas. Por
### fim, classificamos os dados e printamos as métricas.

#### RELATÓRIO (tarefa 8)
## Para cada um dos dez datasets realizamos o mesmo processo. Inicialmente carregamos cada deles um separando em dois
## arrays: features e classes. Após carregar os dados, passamos para a função "use_data" de utils, que realiza todas
## as tarefas do trabalho. Inicialmente, pre-processamos os dados com a função "pre_process_data", (tarefa 1) em
## seguida buscamos as envoltórias convexas com "create_convex_hull", que utiliza o algoritmo de chan contido em chan.py,
## para encontrar a envoltória. (tarefa 2) Depois rodamos o algoritmo de varredura linear implementado em
## "envelop_convex_overlap" para verificar se há sobreposição entre as envoltórias, (tarefa 3, 4) em seguida encontramos
## a linha entre as envoltórias com "find_line_between_hulls", (tarefa 5) depois classificamos os dados de teste e por
## fim (tarefa 6, 7) verificamos e printamos no terminal as métricas pedidas.
## Note, ainda, que dentro de "find_line_between_hulls" chamamos "plot_everything", que como o nome diz, plota as
## envoltórias e as retas.

## Por Ricardo Dias Avelar - 2019054960

# 1
iris_data = load_iris()
utils.use_data(iris_data.data, iris_data.target, True)

# 2
X, y = fetch_covtype(return_X_y=True)
utils.use_data(X, y, False)

# 3
digits_data = load_digits()
utils.use_data(digits_data.data, digits_data.target, True)

# 4
wine_data = load_wine()
utils.use_data(wine_data.data, wine_data.target, True)

# 5
bc_data = load_breast_cancer()
utils.use_data(bc_data.data, bc_data.target, True)

# 6
fof_data = fetch_olivetti_faces()
utils.use_data(fof_data.data, fof_data.target, True)

# 7
flp_data = fetch_lfw_people()
utils.use_data(flp_data.data, flp_data.target, True)

# 8
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
glass_columns = ['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
glass_data = pd.read_csv(url, names=glass_columns)
X = glass_data.drop(['ID', 'Type'], axis=1).values
y = glass_data['Type'].values
utils.use_data(X, y, True)

# 9
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = np.array(y_train)
y_test = np.array(y_test)
X = np.vstack((x_train, x_test))[0]
y = np.vstack((y_train, y_test)).flatten()
utils.use_data(X, y, True)

# 10
url3 = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
         'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
dataset = pd.read_csv(url3, names=names)
X = dataset.iloc[:, :-1].values
X = np.vectorize(utils.convert_to_float)(X)
y = dataset.iloc[:, -1].values
utils.use_data(X, y, True)
