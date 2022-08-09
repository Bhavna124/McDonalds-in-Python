# McDonalds-in-Python
import numpy as np
import pandas as pd
import io 
import requests
c.columns
import seaborn as sns
url="https://homepage.boku.ac.at/leisch/MSA/datasets/mcdonalds.csv"
s=requests.get(url).content  
c=pd.read_csv(io.StringIO(s.decode('utf-8')))
print(c)
c.columns
c.ndim
c.shape
c.head()
c['yummy'].replace(['Yes','No'], [0,1],inplace=True)
c['convenient'].replace(['Yes','No'], [0,1],inplace=True)
c['spicy'].replace(['Yes','No'], [0,1],inplace=True)
c['fattening'].replace(['Yes','No'], [0,1],inplace=True)
c['greasy'].replace(['Yes','No'], [0,1],inplace=True)
c['fast'].replace(['Yes','No'], [0,1],inplace=True)
c['cheap'].replace(['Yes','No'], [0,1],inplace=True)
c['tasty'].replace(['Yes','No'], [0,1],inplace=True)
c['expensive'].replace(['Yes','No'], [0,1],inplace=True)
c['healthy'].replace(['Yes','No'], [0,1],inplace=True)
c['disgusting'].replace(['Yes','No'], [0,1],inplace=True)
print(c)
y=c.iloc[:,1:11].values
print(y)
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(y)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pca1', 'pca2','pca3','pca4','pca5','pca6','pca7','pca8','pca9','pca10'])
print(principalDf)
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit_transform(y)
plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    )
plt.xlabel('Features of PCA')
plt.ylabel('Variance is explained')
plt.title('Variance is explained by Features')
plt.show()
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.decomposition import PCA
sns.set()
pca = PCA(n_components=10)
reduced_features = pca.fit_transform(y)
plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    ) 
plt.plot(
    range(1,len(pca.explained_variance_ )+1),
    np.cumsum(pca.explained_variance_),
    c='red',
    label='Cumulative Explained Variance')
plt.legend(loc='upper left')
plt.xlabel('Number of components')
plt.ylabel('Explained variance (eignenvalues)')
plt.title('Plot')
plt.show()
loadings = pca.components_
import matplotlib.pyplot as plt 
import numpy as np
x = loadings[0]
y = loadings[1]
for i, varnames in enumerate(c):
    plt.scatter(x[i], y[i], s=200)
    plt.text(x[i], y[i], varnames)
xticks = np.linspace(-0.8, 0.8, num=5)
yticks = np.linspace(-0.8, 0.8, num=5)
plt.xticks(xticks)
plt.yticks(yticks)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('2D Loading plot')
pca_df_scaled = principalDf 
scaler_df = principalDf[['pca1', 'pca2','pca3','pca4','pca5','pca6','pca7','pca8','pca9','pca10']]
scaler = 1 / (scaler_df.max() - scaler_df.min())
for index in scaler.index:
    pca_df_scaled[index] *= scaler[index]
pca_df_scaled
