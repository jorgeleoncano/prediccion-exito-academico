import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # necesario para guardar figuras desde funciones sin abrir ventanas
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from limpieza import cargar_datos, limpiar_base

seed = 42  # fijamos la semilla para que los resultados sean siempre los mismos

# variables de rendimiento + perfil básico — las categóricas de alta cardinalidad no ayudan a definir grupos
VARIABLES = [
    "nota_admision",
    "nota_media_1sem",
    "nota_media_2sem",
    "tasa_aprobacion_1sem",
    "tasa_aprobacion_2sem",
    "asignaturas_1sem_aprobadas",
    "asignaturas_2sem_aprobadas",
    "edad_al_matricularse",
    "becado",
    "deudor",
    "matricula_al_dia"]


def limpiar_y_preparar(data):
    data = limpiar_base(data)

    # tasa del 2do semestre: especifica de no supervisado (en regresion no existe por leakage)
    data["tasa_aprobacion_2sem"] = data["asignaturas_2sem_aprobadas"] / data["asignaturas_2sem_evaluadas"].replace(0, 1)

    # las variables binarias estan guardadas como texto ('si'/'no'), las convertimos a 0/1
    for col in ["becado", "deudor", "matricula_al_dia"]:
        data[col] = data[col].map({"si": 1, "no": 0})

    # nos quedamos solo con las variables elegidas
    X = data[VARIABLES].copy()
    objetivo = data["objetivo"].copy()

    # escalamos los datos: obligatorio para PCA y K-Means
    # sin escalar, nota_admision (0-200) dominaría sobre becado (0/1) y los resultados serían una basura
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Datos tras limpieza: {len(data)} estudiantes")
    print(f"Variables seleccionadas: {len(VARIABLES)}")

    return X_scaled, X, objetivo


def analizar_pca(X_scaled):
    # ajustamos PCA completo para ver cuánta varianza captura cada componente
    pca_full = PCA(random_state=seed)
    pca_full.fit(X_scaled)

    varianza_acumulada = np.cumsum(pca_full.explained_variance_ratio_)
    n_para_80 = int(np.argmax(varianza_acumulada >= 0.80)) + 1

    print(f"\nPCA - varianza explicada:")
    print(f"  PC1: {pca_full.explained_variance_ratio_[0]*100:.1f}%")
    print(f"  PC2: {pca_full.explained_variance_ratio_[1]*100:.1f}%")
    print(f"  Componentes necesarias para llegar al 80%: {n_para_80}")

    # scree plot (varianza por componente) y varianza acumulada
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    n = len(pca_full.explained_variance_ratio_)
    axes[0].bar(range(1, n + 1), pca_full.explained_variance_ratio_ * 100, color="#3498db", alpha=0.85)
    axes[0].set_xlabel("Componente principal")
    axes[0].set_ylabel("Varianza explicada (%)")
    axes[0].set_title("Varianza explicada por componente (scree plot)")

    axes[1].plot(range(1, n + 1), varianza_acumulada * 100, "o-", color="#e74c3c", linewidth=2)
    axes[1].axhline(80, linestyle="--", color="gray", linewidth=1, label="80%")
    axes[1].set_xlabel("Numero de componentes")
    axes[1].set_ylabel("Varianza acumulada (%)")
    axes[1].set_title("Varianza acumulada")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("fig_pca_varianza.png", dpi=150)
    plt.close()

    # ahora ajustamos PCA con solo 2 componentes para poder visualizar en 2D
    pca_2d = PCA(n_components=2, random_state=seed)
    X_pca = pca_2d.fit_transform(X_scaled)

    # biplot: cada flecha muestra hacia dónde "apunta" cada variable en el espacio PCA
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.12, s=8, color="#95a5a6")

    loadings = pca_2d.components_.T  # cada fila es una variable, cada columna es una PC
    scale = 3.5  # factor para que las flechas del biplot no se queden pegadas al centro
    for i, var in enumerate(VARIABLES):
        ax.annotate("", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))
        ax.text(loadings[i, 0] * scale * 1.12, loadings[i, 1] * scale * 1.12, var, fontsize=8, color="#c0392b")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Biplot PCA — contribucion de cada variable a las componentes principales")
    plt.tight_layout()
    plt.savefig("fig_pca_biplot.png", dpi=150)
    plt.close()

    return X_pca, pca_full


def elegir_numero_clusters(X_scaled):
    # probamos k de 2 a 8: la inercia siempre baja, buscamos el "codo" donde ya no merece la pena
    # el silhouette mide si los clusters están bien separados (más cercano a 1 = mejor)
    rango_k = range(2, 9)
    inercias = []
    silhouettes = []

    for k in rango_k:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        etiquetas = km.fit_predict(X_scaled)
        inercias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, etiquetas))

    # gráfico del codo y silhouette
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(rango_k, inercias, "o-", color="#3498db", linewidth=2)
    axes[0].set_xlabel("Numero de clusters (k)")
    axes[0].set_ylabel("Inercia")
    axes[0].set_title("Metodo del codo")

    axes[1].plot(rango_k, silhouettes, "o-", color="#2ecc71", linewidth=2)
    axes[1].set_xlabel("Numero de clusters (k)")
    axes[1].set_ylabel("Silhouette")
    axes[1].set_title("Coeficiente de silhouette")

    plt.tight_layout()
    plt.savefig("fig_codo_silhouette.png", dpi=150)
    plt.close()

    print("\nBUSQUEDA DE K OPTIMO")
    filas = []
    for i, k in enumerate(rango_k):
        filas.append({"k": k, "Inercia": round(inercias[i], 1), "Silhouette": round(silhouettes[i], 4)})
    print(pd.DataFrame(filas).to_string(index=False))

    # devolvemos el k con mejor silhouette
    mejor_k = list(rango_k)[int(np.argmax(silhouettes))]
    print(f"\nMejor k segun silhouette: {mejor_k}")
    return mejor_k


def clustering_kmeans(X_scaled, X_pca, k):
    # entrenamos K-Means con el k elegido y asignamos cada estudiante a un cluster
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    etiquetas = km.fit_predict(X_scaled)

    # pintamos los clusters en el espacio PCA
    colores = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]  # un color distinto por cluster
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(k):
        mask = etiquetas == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.4, s=10,
                   color=colores[i], label=f"Cluster {i + 1}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Clusters K-Means (k={k}) en espacio PCA")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig_clusters_pca.png", dpi=150)
    plt.close()

    return etiquetas


def visualizar_vs_objetivo(X_pca, objetivo):
    # los mismos estudiantes pero con su clase real, para comparar con los clusters
    mapa_colores = {"abandono": "#e74c3c", "matriculado": "#f39c12", "graduado": "#2ecc71"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for clase, color in mapa_colores.items():
        mask = objetivo.values == clase  # seleccionamos solo los estudiantes de esta clase
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.4, s=10, color=color, label=clase)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Clases reales (objetivo) en espacio PCA")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig_objetivo_pca.png", dpi=150)
    plt.close()


def perfilar_clusters(X, etiquetas, objetivo):
    # media de cada variable por cluster y qué clases reales hay en cada uno
    X_copia = X.copy()
    X_copia["cluster"] = etiquetas
    X_copia["objetivo"] = objetivo.values

    print(f"\nPERFIL MEDIO DE CADA CLUSTER")
    perfil = X_copia.groupby("cluster")[VARIABLES].mean().round(3)
    print(perfil.T.to_string())  # .T transpone: variables en filas y clusters en columnas, más fácil de leer

    print(f"\nDISTRIBUCION DE CLASES POR CLUSTER (%)")
    # crosstab con normalize="index" da el % de cada clase dentro de cada cluster (por filas)
    dist = (pd.crosstab(X_copia["cluster"], X_copia["objetivo"], normalize="index") * 100).round(1)
    print(dist.to_string())



if __name__ == "__main__":
    data = cargar_datos()
    X_scaled, X, objetivo = limpiar_y_preparar(data)

    X_pca, pca_full = analizar_pca(X_scaled)

    mejor_k = elegir_numero_clusters(X_scaled)

    etiquetas = clustering_kmeans(X_scaled, X_pca, mejor_k)

    visualizar_vs_objetivo(X_pca, objetivo)

    perfilar_clusters(X, etiquetas, objetivo)

    print("\nAprendizaje no supervisado completado.")
