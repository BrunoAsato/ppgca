import numpy as np
import pandas as pd
import statistics as stcs
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
# Teste de biblioteca para gráficos
import matplotlib.pyplot as plt
# Biblioteca para salvar e carregar modelo
import pickle
from datetime import datetime
#from sklearn.datasets.samples_generator import make_blobs


# Variável com o nome da base de dados
filename = 'ValoresPontosVoo1.csv'
# Imprime um aviso de carregamento da base
print("-> Carregando a base de dados %s." % filename)
# Le os dados do arquivo CSV
dataset = pd.read_csv(filename, delimiter=';')

# Obtem os valores de R, G, NIR e o valor de PAR de todos os registros
xs = dataset.iloc[:, [1,2,3]].values.astype(float)
# Obtem somente a coluna de PAR
ys = dataset.iloc[:, 4]
# Atribui os valores de linhas x colunas as variaveis n_samples, n_instances
n_samples, n_instances = xs.shape
# Imprime os números de linhas x colunas
print("-> Feito. Shape =", n_samples, " x ", n_instances)
r_scores = []
vscores = []
for i in range(10):
	# Gera um valor aleatório 
	rng = np.random.RandomState(datetime.now().microsecond)
	# Divide a base em teste e treino 
	x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size = 0.3, random_state = rng)
	
	# Executa a Regressão Bayesiana
	#reg = linear_model.LogisticRegression()
	#reg = neural_network.MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
	reg = svm.SVR(kernel='sigmoid', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
	#reg = linear_model.LogisticRegression()
	
	# Gera o modelo
	reg.fit(x_train, y_train)
	# Gravar o modelo em arquivo
	#filenamesave = 'BayesianRidge.sav'
	#pickle.dump(reg, open(filenamesave, 'wb'))

	# Carregar modelo salvo
	#loaded_model = pickle.load(open(filename, 'rb'))
	#result = loaded_model.score(X_test, Y_test)
	#print(result)
	# Testa o modelo gerado
	pred = reg.predict(x_test)
	# Armazena os resultados
	vscores.append(reg.score(x_test, y_test))
	r_scores.append(r2_score(y_test, pred))
	
	#print('R score =', corrcoef(x_test, y_test))
	print('R score =', reg.score(x_test, y_test))
	print('R2 score =', r2_score(y_test, pred))
	print('-----------------')
vscores=pd.DataFrame(vscores).values.astype(float)
r_scores=pd.DataFrame(r_scores).values.astype(float)

print('Min =', vscores.min())
print('Max =', vscores.max())
print('Avg =', vscores.mean())
print('Desvio padrão =', vscores.std())

	#print(reg.coef_)
# Plot outputs
#plt.scatter(x_test, y_test,  color='gray')
#plt.plot(x_test, pred, color='red', linewidth=2)
#plt.show()

#Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
# make a prediction
#ynew = model.predict_proba(Xnew)
# show the inputs and predicted probabilities
#for i in range(len(Xnew)):
#	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
