#
# This code is intentionally missing!
# Read the directions on the course lab page!
#

import pandas as pd
import sklearn.preprocessing as pp
X = pd.read_csv('Datasets/parkinsons.data')
y = X['status']
X.drop(['name','status'],axis=1,inplace=True)

import sklearn.model_selection as ms
X_trainI,X_testI,y_train,y_test = ms.train_test_split(X,y,test_size= 0.3,random_state=7)
norma = pp.StandardScaler().fit(X_trainI); X_trainI = norma.transform(X_trainI); X_testI = norma.transform(X_testI);
# StandardScaler best

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
best_score = 0

for nb_dim in range(2,6,1): 
    iso = Isomap(n_neighbors=nb_dim,n_components=6).fit(X_trainI)
    X_train = iso.transform(X_trainI)
    X_test = iso.transform(X_testI)    
    for C in range(5,205,5):
        for gamma in range(1,101,1):
            svc = SVC(C=C/100,gamma=gamma/1000).fit(X_train,y_train)
            score = svc.score(X_test,y_test)
            if score > best_score: best_score = score
        
    
print(best_score)
            