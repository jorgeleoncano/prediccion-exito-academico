import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # necesario para guardar figuras desde funciones sin abrir ventanas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from limpieza import cargar_datos, limpiar_base
from cv import buscar_mejores_modelos_regresion

seed = 42  # fijamos la semilla para que los resultados sean siempre los mismos


def limpiar_datos(data):
    data = limpiar_base(data)

    # eliminamos las variables del 2do semestre (excepto nota_media_2sem que es nuestro objetivo)
    # si las dejáramos, el modelo aprendería a hacer trampa: saber cuántas asignaturas aprobó
    # en el 2do semestre ya implica saber directamente la nota del 2do semestre
    cols_2sem = [
        "asignaturas_2sem_convalidadas",
        "asignaturas_2sem_matriculadas",
        "asignaturas_2sem_evaluadas",
        "asignaturas_2sem_aprobadas",
        "asignaturas_2sem_sin_evaluacion",
    ]
    data = data.drop(columns=cols_2sem)

    # eliminamos el objetivo de clasificacion: saber si un estudiante graduó o abandonó
    # implica conocer su rendimiento final, que incluye el 2do semestre → es leakage
    data = data.drop(columns=["objetivo"])

    # quitamos los estudiantes con nota_media_2sem == 0 porque son estudiantes que no llegaron
    # a presentarse al 2do semestre y distorsionan el modelo (queremos predecir notas reales)
    data = data[data["nota_media_2sem"] > 0]

    print(f"Datos tras limpieza: {data.shape[0]} estudiantes, {data.shape[1]} variables")
    return data


def preprocesar(data):
    # separamos X (variables de entrada) e y (lo que queremos predecir)
    X = data.drop("nota_media_2sem", axis=1)
    y = data["nota_media_2sem"]

    # identificamos que columnas son texto (categoricas) y cuales son numeros
    cat_cols = X.select_dtypes(include="str").columns.tolist()
    num_cols = X.select_dtypes(exclude="str").columns.tolist()

    # convertimos las categoricas a numeros con one hot encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_encoded = encoder.fit_transform(X[cat_cols])

    # guardamos los nombres para la gráfica de importancias
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    feature_names = list(cat_feature_names) + num_cols

    # juntamos las columnas codificadas con las que ya eran numéricas en una sola matriz
    X_encoded = np.hstack([X_cat_encoded, X[num_cols].values])

    # 80% train y 20% test
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y.values, test_size=0.2, random_state=seed)

    # escalamos los datos solo para los modelos lineales (regresion lineal y ridge)
    # los arboles no lo necesitan porque no tienen en cuenta la escala de las variables
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Preprocesado terminado:")
    print(f"\nTenemos {len(X_train)} muestras en train (80%)")
    print(f"Tenemos {len(X_test)} muestras en test (20%)")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names



def validacion_cruzada(modelos, X_train_por_modelo, y_train):
    # cross validation con 5 particiones antes de ir al test
    # RMSE como métrica principal (mismas unidades que la nota) y R2 para ver cuánto explica el modelo
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    resultados_cv = {}
    filas = []

    for nombre, modelo in modelos.items():
        scores = cross_validate(modelo, X_train_por_modelo[nombre], y_train, cv=cv,scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"}, n_jobs=-1)
        resultados_cv[nombre] = scores
        filas.append({
            "Modelo": nombre,
            "RMSE": round(-scores["test_rmse"].mean(), 4),
            "+-": round(scores["test_rmse"].std(), 4),
            "R2": round(scores["test_r2"].mean(), 4),})

    print("\nVALIDACION CRUZADA KFOLD (k=5)")
    print(pd.DataFrame(filas).to_string(index=False))

    return resultados_cv


def evaluar_en_test(modelos, resultados_cv, X_train_por_modelo, X_test, X_test_scaled, y_train, y_test):
    # elegimos el mejor modelo segun el RMSE de la validacion cruzada (el que tiene el RMSE mas bajo)
    mejor = None
    mejor_rmse = float("inf")  # empezamos en infinito para que cualquier RMSE real sea menor
    for nombre in resultados_cv:
        rmse = -resultados_cv[nombre]["test_rmse"].mean()
        if rmse < mejor_rmse:
            mejor_rmse = rmse
            mejor = nombre
    print(f"\nMejor modelo (RMSE CV): {mejor}")

    # entrenamos el mejor modelo con todo el train y predecimos en test
    if mejor in ("Regresion Lineal", "Ridge"):
        X_test_final = X_test_scaled
    else:
        X_test_final = X_test

    modelo_final = modelos[mejor]
    modelo_final.fit(X_train_por_modelo[mejor], y_train)
    y_pred = modelo_final.predict(X_test_final)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\nResultados en test:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")

    # scatter plot: valor real vs prediccion
    # si el modelo fuera perfecto, todos los puntos estarían en la diagonal
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="#3498db")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="prediccion perfecta")
    ax.set_xlabel("Nota real")
    ax.set_ylabel("Nota predicha")
    ax.set_title(f"Real vs Predicho ({mejor})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig_real_vs_predicho.png", dpi=150)
    plt.close()

    # tabla resumen con los 4 modelos para poder compararlos facilmente en el informe
    filas = []
    for nombre, modelo in modelos.items():
        X_tr = X_train_por_modelo[nombre]
        if nombre in ("Regresion Lineal", "Ridge"):
            X_te = X_test_scaled
        else:
            X_te = X_test
        modelo.fit(X_tr, y_train)
        yp = modelo.predict(X_te)
        if nombre == mejor:
            etiqueta = nombre + " (mejor)"
        else:
            etiqueta = nombre
        filas.append({
            "Modelo": etiqueta,
            "RMSE": round(np.sqrt(mean_squared_error(y_test, yp)), 4),
            "MAE": round(mean_absolute_error(y_test, yp), 4),
            "R2": round(r2_score(y_test, yp), 4),
        })

    print("\nResumen final en test:")
    print(pd.DataFrame(filas).to_string(index=False))


def importancia_variables(modelos, X_train_por_modelo, y_train, feature_names):
    # con RF podemos saber la importancia de cada variable facilmente
    # hacemos un gráfico con las top 15 mas relevantes y luego imprimimos el top 5
    rf = modelos["Random Forest"]
    rf.fit(X_train_por_modelo["Random Forest"], y_train)
    importancias = pd.Series(rf.feature_importances_, index=feature_names).nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    importancias.plot(kind="barh", ax=ax, color="#3498db", alpha=0.85)
    ax.set_title("Top 15 variables mas importantes (Random Forest Regressor)")
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    plt.savefig("fig_reg_feature_importance.png", dpi=150)
    plt.close()

    print(f"\nTop 5 variables mas importantes:")
    print(importancias.tail(5).to_string())


if __name__ == "__main__":
    data = cargar_datos()
    print(f"\nVariable objetivo: nota_media_2sem")
    print(f"  Media:   {data['nota_media_2sem'].mean():.2f}")
    print(f"  Mediana: {data['nota_media_2sem'].median():.2f}")
    print(f"  Min:     {data['nota_media_2sem'].min():.2f}")
    print(f"  Max:     {data['nota_media_2sem'].max():.2f}")
    data = limpiar_datos(data)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names = preprocesar(data)

    modelos, X_train_por_modelo = buscar_mejores_modelos_regresion(X_train, X_train_scaled, y_train)

    resultados_cv = validacion_cruzada(modelos, X_train_por_modelo, y_train)

    evaluar_en_test(modelos, resultados_cv, X_train_por_modelo, X_test, X_test_scaled, y_train, y_test)

    importancia_variables(modelos, X_train_por_modelo, y_train, feature_names)

    print("\nRegresion completada.")
