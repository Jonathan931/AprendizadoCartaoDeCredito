#Base 
import numpy as np;
import pandas as pd;
from sklearn import svm;

def carregarDados():
    dados = pd.read_csv('./dataset/creditcard.csv');

    x = dados.drop('Class', axis=1)
    y = dados.Class

    return x, y;


def main():
    dados, resultados = carregarDados();

    print( "Dados carregados...." )

    from sklearn.model_selection import train_test_split
    treino_dados, teste_dados, treino_resultados, teste_resultados = train_test_split(dados, resultados, test_size=0.1)

    from sklearn.tree import DecisionTreeClassifier
    #min_samples_leaf - Quantidade mínima de amostras necessárias para ser uma folha
    #min_samples_split - Quantidade mínima de amostras necessárias para dividir um nó
    clf = DecisionTreeClassifier(random_state=0, min_samples_leaf= 20, min_samples_split=50)

    print("Iniciar treino....")

    model = clf.fit(treino_dados, treino_resultados)

    print("Treino concluido....")

    valor = model.score(teste_dados, teste_resultados)
    print( "Precisão por Árvore de Decisão: ", valor )


if __name__ == '__main__': # chamada da funcao principal
    main() # chamada da função main