import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data_dict = pd.read_pickle('/Users/marcchkeiban/Desktop/courses/Deep learning/Project/Datas/large_cap_x_data.pickle')

# Get the first key (timestamp) and its corresponding DataFrame
first_date = list(data_dict.keys())[0]
for i in range(len(data_dict)):
    df = data_dict[i]
    pca = PCA(n_components=8)
    df_pca = pca.fit_transform(df)
    print(df_pca.shape)
    print(df.shape)

df = data_dict[first_date]

print(f"Data is a dictionary with {len(data_dict)} dates.")
print(f"Showing data for the first date: {first_date}")
print(df.head())

'''for i in range(2, 16):
    agg = AgglomerativeClustering(n_clusters=i)
    agg.fit(df)
    print(f"Silhouette score for {i} clusters: {silhouette_score(df, agg.labels_)}")'''

pca = PCA(n_components=8)
pca.fit(df)
#print(f"PCA score for {i} components: {pca.explained_variance_ratio_.sum()}")

new_df = pca.transform(df)
print(new_df.shape)
print(df.shape)