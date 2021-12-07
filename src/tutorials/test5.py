import pandas as pd
import os
from ivpy import attach, show, montage, histogram, scatter, compose
from ivpy.extract import extract
from ivpy.reduce import pca, tsne, umap
from ivpy.cluster import cluster
from ivpy.extract import norm
from sklearn.metrics import adjusted_rand_score as adjrand
import tqdm

DIR = "/mnt/e/Tasks/similarity/Images/Situated-Views-Lowres/"
df = pd.read_csv("/mnt/e/Tasks/similarity/src/output/metadata.csv")
df = df.loc[
    :,
    [
        "Source ID",
        "Creator",
        "First Year",
        "Type",
        "Width (mm)",
        "Height (mm)",
        "Latitude",
        "Longitude",
        "entropy",
        "brightness",
        "contrast",
        "homogeneity",
        "creators_number",
        "path",
    ],
]


L = os.listdir(DIR)
print(L)
attach(df, "path")

print("monting teste 3..")
# montage(xcol="entropy", shape="circle", ascending=True).save(
#     "/mnt/e/Tasks/similarity/src/plots/teste3.png"
# )
print("extrating...")

X = extract("neural")
# df["neural"] = X
X = norm(X)

# print("Saving metadata_neural..")
# df.to_csv("/mnt/e/Tasks/similarity/src/output/metadata_neural.csv")

df["cluster_kmeans_20"] = cluster(X, k=20)
print("monting teste 4..")
montage(facetcol="cluster_kmeans_20").save(
    "/mnt/e/Tasks/similarity/src/plots/teste4.png"
)


df["cluster_kmeans"] = cluster(X, k=20)

print("Saving teste 5..")
montage(facetcol="cluster_kmeans_20").save(
    "/mnt/e/Tasks/similarity/src/plots/teste5.png"
)

d = dict(zip(list(df.Creator.unique()), list(range(len(df.Creator.unique())))))
df["Creator_number"] = [d[item] for item in df.Creator]
adjrand(df.Creator_number, df.cluster_kmeans)

for cluster_number in df.cluster_kmeans.unique():
    tmp = df.Creator[df.cluster_kmeans == cluster_number]
    n = len(tmp.unique())
    print(cluster_number, ":", n)


print("Saving teste 6..")
montage(
    pathcol=df.filename[df.cluster_kmeans == 3],
    notecol=df.Creator[df.cluster_kmeans == 3],
).save("/mnt/e/Tasks/similarity/src/plots/teste6.png")


plotlist = []
for func in [pca, tsne, umap]:
    df[["x", "y"]] = func(X, n_components=2)
    plotlist.append(scatter("x", "y", side=800, xbins=40, ybins=40, thumb=20))

print("Saving teste 7..")
compose(*plotlist, ncols=3, border=True).save(
    "/mnt/e/Tasks/similarity/src/plots/teste7.png"
)

print("Saving teste 8..")
df[["x", "y"]] = tsne(X)
scatter("x", "y", side=4000, thumb=64).save(
    "/mnt/e/Tasks/similarity/src/plots/teste8.png"
)
