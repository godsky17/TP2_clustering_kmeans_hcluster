import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clustering K-means/Hclust", layout="wide")

st.title("TP2 – Clustering K-means / Hclust")

# 1-a : Chargement des données
st.header("Chargement des données")
uploaded_file = st.file_uploader("Charger un fichier .csv", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données :", data.head())
    numeric_data = data.select_dtypes(include=['int64', 'float64'])

    if numeric_data.shape[1] < 2:
        st.warning("Le fichier doit contenir au moins 2 colonnes numériques.")
    else:
        # 1-b : Choix de K
        st.header("Choix de K")
        k = st.slider("Choisissez le nombre de clusters K :", min_value=2, max_value=10, value=3)

        # 1-c : K-means
        st.header("1-c. Résultats du K-means")
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(numeric_data)
        data["Cluster"] = clusters

        st.subheader("Centroïdes")
        st.write(pd.DataFrame(kmeans.cluster_centers_, columns=numeric_data.columns))

        st.subheader("Écart-types des clusters")
        st.write(data.groupby("Cluster")[numeric_data.columns].std())

        # Dendrogramme (Hclust)
        st.subheader("Dendrogramme (Hclust)")
        Z = linkage(numeric_data, method='ward')
        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(Z, ax=ax)
        st.pyplot(fig)

        # 1-d : Visualisation (PCA)
        st.header("Visualisation des clusters (PCA)")
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(numeric_data)
        reduced_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
        reduced_df["Cluster"] = clusters

        fig2, ax2 = plt.subplots()
        for cluster_id in range(k):
            cluster_points = reduced_df[reduced_df["Cluster"] == cluster_id]
            ax2.scatter(cluster_points["PC1"], cluster_points["PC2"], label=f"Cluster {cluster_id}")
        ax2.legend()
        st.pyplot(fig2)

        # 1-e : Prédiction d’un point
        st.header("Prédiction d’un point")
        st.write("Entrez les valeurs pour un point à prédire :")
        inputs = []
        for col in numeric_data.columns:
            val = st.number_input(f"{col}", value=float(numeric_data[col].mean()))
            inputs.append(val)
        if st.button("Prédire le cluster"):
            prediction = kmeans.predict([inputs])[0]
            st.success(f"Ce point appartient au cluster {prediction}")

        # 1-f : Métriques de qualité
        st.header("Métriques de qualité")
        inertia = kmeans.inertia_
        silhouette = silhouette_score(numeric_data, clusters)
        st.write(f"Inertie (intra-cluster) : {inertia:.2f}")
        st.write(f"Score de silhouette (inter-cluster) : {silhouette:.2f}")

        # 1-g : Bonus – Suggestion de K
        st.header("Suggestion d’un K optimal")
        k_range = list(range(2, 11))
        inertias = []
        silhouettes = []

        for k_val in k_range:
            model = KMeans(n_clusters=k_val, random_state=42)
            preds = model.fit_predict(numeric_data)
            inertias.append(model.inertia_)
            silhouettes.append(silhouette_score(numeric_data, preds))

        fig3, ax3 = plt.subplots()
        ax3.plot(k_range, inertias, marker='o', label="Inertie")
        ax3.set_ylabel("Inertie")
        ax3.set_xlabel("K")
        ax3.set_title("Méthode du coude")
        st.pyplot(fig3)

        fig4, ax4 = plt.subplots()
        ax4.plot(k_range, silhouettes, marker='x', color='green', label="Silhouette")
        ax4.set_ylabel("Score de silhouette")
        ax4.set_xlabel("K")
        ax4.set_title("Score de silhouette par K")
        st.pyplot(fig4)

        optimal_k = k_range[np.argmax(silhouettes)]
        st.success(f"K suggéré selon silhouette : {optimal_k}")

else:
    st.info("Veuillez charger un fichier CSV pour commencer.")
