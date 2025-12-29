import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Plateforme Data Mining", layout="wide")

st.title("📊 Plateforme de Data Mining No‑Code")
st.write("Application complète : EDA, Prétraitement, KNN et K‑Means")

# =========================
# 1. Chargement des données
# =========================
st.header("1️⃣ Chargement du fichier CSV")

uploaded_file = st.file_uploader(
    "📂 Chargez votre fichier CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Fichier chargé avec succès")

    # =========================
    # 2. EDA
    # =========================
    st.header("2️⃣ Exploration des données (EDA)")

    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    st.subheader("Statistiques descriptives")
    st.write(df.describe())

    # =========================
    # 3. Prétraitement
    # =========================
    st.header("3️⃣ Prétraitement des données")

    features = ["math", "physics", "computer_science"]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_scaled = pd.DataFrame(X_scaled, columns=features)
    st.write("Aperçu des données après normalisation")
    st.dataframe(df_scaled.head())

    # =========================
    # 4. Classification KNN
    # =========================
    st.header("4️⃣ Classification – KNN")

    df["resultat"] = df["average"].apply(
        lambda x: "Réussite" if x >= 10 else "Échec"
    )

    y = df["resultat"]

    X_train, X_test, y_train, y_test = train_test_split(
        df_scaled, y, test_size=0.2, random_state=42
    )

    k_knn = st.slider("Choisissez la valeur de K (KNN)", 1, 15, 5)

    knn = KNeighborsClassifier(n_neighbors=k_knn)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    st.write("Accuracy :", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=["Réussite", "Échec"])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Réussite", "Échec"]
    )

    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

    # =========================
    # 5. Clustering K-Means
    # =========================
    st.header("5️⃣ Clustering – K‑Means")

    k_cluster = st.slider("Choisissez le nombre de clusters (K‑Means)", 2, 6, 3)

    kmeans = KMeans(n_clusters=k_cluster, random_state=42)
    df["cluster"] = kmeans.fit_predict(df_scaled)

    st.subheader("Données avec clusters")
    st.dataframe(df.head())

    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(
        df["math"],
        df["physics"],
        c=df["cluster"],
        cmap="viridis"
    )
    ax2.set_xlabel("Math")
    ax2.set_ylabel("Physics")
    plt.colorbar(scatter, label="Cluster")
    st.pyplot(fig2)
