import numpy as np
from sklearn import preprocessing, neighbors,neural_network
from sklearn.neural_network import MLPClassifier
import pandas as pd
import statistics
import pickle
import matplotlib.pyplot as plt
def get_labels(X):
    ysContainer = []
    for xs in X:
        data_req = statistics.mean(xs)
        if data_req > 5:
            ysContainer.append(4)
        else:
            ysContainer.append(2)
    return ysContainer
df = pd.read_csv('Data\gf_data.csv')
df2 = pd.read_csv('Data\data_test.csv')
df.drop(['gf_id'],1,inplace=True)
df2.drop(['gf_id'],1,inplace=True)
X = np.array(df.drop(['class'],1))
y = np.array(get_labels(X))
X_test = np.array(df2.drop(['class'],1))
y_test = np.array(get_labels(X_test))
print(y_test)
clf = neighbors.KNeighborsClassifier()
clf.fit(X,y)
nn = MLPClassifier(activation="logistic",batch_size=4,learning_rate_init=0.01,solver="sgd",max_iter=100,shuffle=True)
model_in = open('model.ptl','rb')
# nn = pickle.load(model_in)
losses = []
nn.fit(X,y)
nnac = nn.score(X_test,y_test)
nn.n_outputs_ = 10
print(nn.n_outputs_)
accuracy = clf.score(X_test,y_test)
print("Acurracy: " + str(nnac - nn.loss_))

ex = np.array([10.,10.,8.,10.])
print(statistics.mean(ex))
ex =  ex.reshape(1,-1)
prediction = clf.predict(ex)
print(prediction)
print(nn.predict(ex))


# with open('model.ptl','wb') as f:
#     pickle.dump(nn,f)