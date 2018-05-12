#Base 
import numpy as np;
import pandas as pd;
from sklearn import svm;

def carregarDados():
	dados = pd.read_csv('./dataset/creditcard.csv');
	x = dados[
	['Time',  'V1',  'V2',  'V3', 'V4',
	   'V5',  'V6',  'V7',  'V8', 'V9', 
	  'V10', 'V11', 'V12', 'V13', 'V14',
	  'V15', 'V16', 'V17', 'V18', 'V19', 
	  'V20', 'V21', 'V22', 'V23', 'V24',
	  'V25', 'V26', 'V27', 'V28', 'Amount']];
	y = dados[['Class']];
	dados = pd.get_dummies(x);
	resultados = pd.get_dummies(y);
	return dados.values.ravel(), resultados.values.ravel();

def main():
	dados, resultados = carregarDados();
	clf = svm.SVC();
	clf.fit(dados, resultados);
	SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

main();
