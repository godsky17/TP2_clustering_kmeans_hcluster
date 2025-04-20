import streamlit as st
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import math

st.set_page_config("TP2 - clustering K-means/Hclust")

# Recuperation de la data
uf = st.file_uploader("Importer votre fichier de donnees (.csv uniquement)", type="csv")
if uf:
    data = pd.read_csv(uf)
    data = data.select_dtypes(include=['float64', 'int64'])

# Choix du K
if not data.empty:
    with st.form("my_form"):
        st.write("Choisir K")
        k = st.slider('K', 1, 10)
        submit = st.form_submit_button('Commencer')
    
    if submit:
        st.write("La valeur de k est : ", k)

    # Determination aleatoire des centroides
    data = data.to_numpy()
    centroides = []
    centroid_index = np.random.randint(1, len(data), size=k)
    for i in range(len(centroid_index)):
        centroides.append(data[centroid_index[i]])
    st.write("Centroides initial :")
    df_centroides = pd.DataFrame(centroides)
    df_centroides.index = [f"centroide {i +1}" for i in range(len(centroides))]
    st.table(df_centroides)
    
def dist(a, b):
    result = 0
    if len(a) == len(b):
        for i in range(len(a)):
            result += pow(a[i] - b[i], 2)
    return math.sqrt(result)

def calculDist(centroide, data):
    tab_dist = []
    for i in range(len(centroide)):
        tab = []
        for j in range(len(data)):
            tab.append(dist(centroide[i], data[j]))
        tab_dist.append(tab)
    return tab_dist
 
# ITTERATION 1
tab_dist_cent_point = calculDist(centroides, data)
df_distances = pd.DataFrame(tab_dist_cent_point).T  # <--- le .T ici est pour la transposition
df_distances.index = [f"Point {j + 1}" for j in range(len(data))]
df_distances.columns = [f"Centroïde {i + 1}" for i in range(len(centroides))]
st.dataframe(df_distances)

# Affectation au cluster
affectation_clusters = df_distances.idxmin(axis=1)
df_affectation = pd.DataFrame({
    "Point": df_distances.index,
    "Cluster": df_distances.idxmin(axis=1)
})

st.dataframe(df_affectation)