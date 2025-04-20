import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

st.title("TP2 – Clustering K-means/Hclust (version légère)")

# 1-a : Chargement CSV
fichier = st.file_uploader("📥 Charger un fichier CSV", type=["csv"])
if fichier:
    data = pd.read_csv(fichier)
    st.write("Aperçu des données :", data.head())
    donnees = data.select_dtypes(include=[np.number])

    if donnees.empty:
        st.error("Le fichier doit contenir des colonnes numériques.")
    else:
        # 1-b : Choix du K
        K = st.number_input("Choisir le nombre de clusters K", min_value=2, max_value=10, value=3)

        # 1-c : K-means basique (sans sklearn)
        def init_centroids(X, k):
            return X.sample(k).to_numpy()

        def assign_clusters(X, centroids):
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            return np.argmin(distances, axis=1)

        def update_centroids(X, labels, k):
            return np.array([X[labels == i].mean(axis=0) for i in range(k)])

        X = donnees.to_numpy()
        centroids = init_centroids(donnees, K)

        for _ in range(10):  # 10 itérations max
            labels = assign_clusters(X, centroids)
            new_centroids = update_centroids(X, labels, K)
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        # Ajout au DataFrame
        data["Cluster"] = labels
        st.write("Centroïdes :")
        st.write(pd.DataFrame(centroids, columns=donnees.columns))

        st.write("Écart-type par cluster :")
        st.write(data.groupby("Cluster")[donnees.columns].mean())

        # 1-c suite : Dendrogramme (Hclust)
        st.subheader("Dendrogramme (Hclust)")
        linkage_matrix = linkage(X, method='ward')
        fig, ax = plt.subplots(figsize=(8, 3))
        dendrogram(linkage_matrix, ax=ax)
        st.pyplot(fig)

        # 1-e : Prédiction d’un point
        st.subheader("Prédiction d’un nouveau point")
        valeurs = []
        for col in donnees.columns:
            val = st.number_input(f"{col} :", value=float(donnees[col].mean()))
            valeurs.append(val)
        if st.button("Prédire le cluster"):
            distances = np.linalg.norm(np.array(valeurs) - centroids, axis=1)
            st.success(f"Ce point appartient au cluster {np.argmin(distances)}")

        # 1-f : Métrique simple = inertie (intra-cluster)
        st.subheader("Inertie totale (cohésion intra-cluster)")
        inertie = sum(np.linalg.norm(X[i] - centroids[labels[i]])**2 for i in range(len(X)))
        st.write(f"Inertie : {inertie:.2f}")
else:
    st.info("Veuillez charger un fichier CSV.")
