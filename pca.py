import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np

cols=["area","perimeter","compactness","length","width","assymetry","groove","class"]
df=pd.read_csv("seeds_dataset.txt",names=cols,sep="\s+")
# print(df.head())
# for i in range(len(cols)-1):
#     for j in range(i+1,len(cols)-1):
#         x_label=cols[i]
#         y_label=cols[j]
#         sns.scatterplot(x=x_label,y=y_label,data=df,hue="class")
#         plt.show()

x="compactness"
y="assymetry"
X=df[[x,y]].values

kmeans=KMeans(n_clusters=3).fit(X)
# clusters=kmeans.labels_
# print(clusters)
# cluster_df=pd.DataFrame(np.hstack((X,clusters.reshape(-1,1))),columns=[x,y,"class"])
# sns.scatterplot(x=x,y=y,hue="class",data=cluster_df)
# plt.show()

# #original df
# sns.scatterplot(x=x,y=y,data=df,hue="class")
# plt.show()

#2D
# X=df[cols[:-1]].values
# kmeans=KMeans(n_clusters=3).fit(X)
# clusters=kmeans.labels_.reshape(-1,1)
# cluster_df=pd.DataFrame(np.hstack((X,clusters)),columns=df.columns)
# sns.scatterplot(x=x,y=y,hue="class",data=cluster_df)
# plt.show()

#PCA
from sklearn.decomposition import PCA
X=df[cols[:-1]].values
pca=PCA(n_components=2)
transformed_x=pca.fit_transform(X)
print(X.shape)
print(transformed_x.shape)
# plt.scatter(transformed_x[:,0],transformed_x[:,1])
# plt.show()
pca_df=pd.DataFrame(np.hstack((transformed_x,kmeans.labels_.reshape(-1,1))),columns=["pca1","pca2","class"])
truth_pca_df=pd.DataFrame(np.hstack((transformed_x,df["class"].values.reshape(-1,1))),columns=["pca1","pca2","class"])
sns.scatterplot(x="pca1",y="pca2",hue="class",data=truth_pca_df)
plt.show()
