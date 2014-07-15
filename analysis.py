#!/usr/bin/env python

import sys
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib
from matplotlib.mlab import PCA


filedir="/Users/snelson/projects/names"
os.chdir(filedir)

years=range(1880, 2013)
pieces=[]
columns=['name', 'sex', 'births']

for year in years:
	path='./yob%d.txt' % year
	frame=pd.read_csv(path, names=columns)
	frame['year']=year
	pieces.append(frame)

names=pd.concat(pieces, ignore_index=True)

total_births=names.pivot_table('births', rows='year', columns='sex', aggfunc=sum)
total_births.head()
total_births.tail()
total_births.plot(title='Total births by sex and year')

def add_prop(group):
	births=group.births.astype(float)
	group['prop']=births/births.sum()
	return group

#check sum
names2=names.groupby(['year', 'sex']).apply(add_prop)
#check=names2.groupby(['year', 'sex']).prop.sum()
#check2=names2.pivot_table('prop', rows='year', columns='sex', aggfunc=sum)
#check2.plot()
#plt.axis([1880, 2020, 0, 2])

def top_1000(group):
	return group.sort_index(by='births', ascending=False)[:1000]

grouped=names2.groupby(['year', 'sex'])
top_1000=grouped.apply(top_1000)

boys=top_1000[top_1000.sex=='M']
girls=top_1000[top_1000.sex=='F']


#plot share of top 1000 names by year/sex
table=top_1000.pivot_table('prop', rows='year', columns='sex', aggfunc=sum)
table.plot()

#plot number of names in top 50%
def get_quantile_count(group, q=0.5):
	group=group.sort_index(by='prop', ascending=False)
	cumsum= np.array(group.prop.cumsum() )
	return cumsum.searchsorted(q)

diversity=top_1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity=diversity.unstack('sex')
diversity.plot(title='Number of popular names in top 50%')

#plot  popularity of these 3 names
births_by_name=boys.pivot_table('prop', rows='year', columns='name', aggfunc=sum).dropna(axis=1)
subset=births_by_name[['Henry', 'Samuel', 'Oscar']]
subset.plot()

#correlations on levels
corr=births_by_name.loc[1960:].pct_change().corr()
corr=corr['Henry'].order(ascending=False)
print corr[corr>0.80]

#plot highly correlated names
fnames=corr.index
births_by_name.loc[:, 'Henry'].pct_change().plot()
births_by_name.loc[:,fnames[:5]].pct_change().plot()

#plot ewma 
ts=births_by_name.pct_change()
pd.ewma(ts[fnames[:3]], span=10).plot()

#lowess plot
fit=pd.DataFrame(index=ts.index[1:])
for name in fnames:
	fit[name]=lowess(exog=ts.index, endog=ts[name])[:,1]

fig, ax=fit[fnames[:10]].plot()

#PCA analysis
p=PCA(fit)
#plot centered data
p.a[:][fnames[0:3]].plot()

#plot first 2 principal components
p1=pd.DataFrame(inde data=p.Y)
p1.loc[:,:2].plot()

#plot fraction of variance explained
plot(p.fracs[:10])

########## cluster analysis ##########
from sklearn import cluster
cl=cluster.KMeans()
cl.n_clusters=4
cluster_assignments=cl.fit_predict(pd.DataFrame.transpose(fit))
plt.plot(cluster_assignments)
plt.show()

glued=pd.DataFrame(index=fnames, data=cluster_assignments, columns=["cluster"])
glued.sort(ascending=False, columns=["cluster"])

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs[0,0].plot(fit[fnames[cluster_assignments!=0]], color='0.75')
axs[0,0].plot(fit[fnames[cluster_assignments==0]],  color='r')
axs[0,0].set_title('Cluster 1')
axs[0,0].set_ylabel('Percent change')
axs[0,1].plot(fit[fnames[cluster_assignments!=1]],  color='0.75')
axs[0,1].plot(fit[fnames[cluster_assignments==1]],  color='r')
axs[0,1].set_title('Cluster 2')
axs[0,1].set_ylabel('Percent change')
axs[1,0].plot(fit[fnames[cluster_assignments!=2]],  color='0.75')
axs[1,0].plot(fit[fnames[cluster_assignments==2]],  color='r')
axs[1,0].set_title('Cluster 3')
axs[1,0].set_ylabel('Percent change')
axs[1,1].set_xlabel('Year')
axs[1,1].plot(fit[fnames[cluster_assignments!=3]],  color='0.75')
axs[1,1].plot(fit[fnames[cluster_assignments==3]],  color='r')
axs[1,1].set_title('Cluster 4')
axs[1,1].set_ylabel('Percent change')
axs[1,1].set_xlabel('Year')
plt.show()














