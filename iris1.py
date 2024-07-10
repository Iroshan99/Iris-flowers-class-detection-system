# IMPORTING LIBRARIES
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

# EXPLORE DATA
print('Dataset Size')
print(dataset.shape)
print('First Five Rows of the Dataset')
print(dataset.head())
print("statisticle summary of the dataset")
print(dataset.describe())
print("get information of the dataset")
print(dataset.info())
print(dataset['class'].value_counts())

sns.violinplot(y='class',x='sepal-lenght',data=dataset,inner='quartile')
plt.show()