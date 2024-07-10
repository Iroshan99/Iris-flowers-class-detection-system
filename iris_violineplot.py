import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
# % matplotlib inline

# LOAD THE DATA
url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
col_name = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = pd.read_csv(url,names=col_name)

sns.violinplot(data=dataset,x='sepal-length',y='class',inner='quartile')
plt.show()
sns.violinplot(data=dataset,x='sepal-width',y='class',inner='quartile')
plt.show()
sns.violinplot(data=dataset,x='petal-length',y='class',inner='quartile')
plt.show()
sns.violinplot(data=dataset,x='petal-width',y='class',inner='quartile')
plt.show()

sns.pairplot(dataset,hue='class',markers='+')
plt.show()

plt.figure(figsize=(7,5))
sns.heatmap(dataset.drop(columns=['class']).corr(), annot=True, cmap='jet')
plt.show()
