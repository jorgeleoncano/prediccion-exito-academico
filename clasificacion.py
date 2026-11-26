import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  #necesario para guardar figuras desde funciones
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from limpieza import cargar_datos, limpiar_base
from cv import buscar_mejores_modelos

seed = 42  # fijamos la semilla para que los resultados sean siempre los mismos


def limpiar_datos(data):
    data = limpiar_base(data)  # quitamos outliers y variables macroeconómicas (ver limpieza.py)
    # tasa del 2do semestre: especifica de clasificacion
    data["tasa_aprobacion_2sem"] = data["asignaturas_2sem_aprobadas"] / data["asignaturas_2sem_evaluadas"].replace(0, 1)
    print(f"Datos tras limpieza: {data.shape[0]} estudiantes, {data.shape[1]} variables")
    return data


def preprocesar(data):
    # separamos lo que el modelo va a usar "x" y lo que va a predecir "y"
    X = data.drop("objetivo", axis=1)
    y = data["objetivo"]

    #vamos a separamos las columnas de texto de las numéricas porque no podemos tratarlas igual
    cat_cols = X.select_dtypes(include="str").columns.tolist()
    num_cols = X.select_dtypes(exclude="str").columns.tolist()
    #convertimos las categóricas con one-hot encoding 
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_encoded = encoder.fit_transform(X[cat_cols])
    #guardamos los nombres para la gráfica de importancias
    feature_names = np.concatenate([encoder.get_feature_names_out(cat_cols), num_cols])
    #juntamos las columnas codificadas con las numéricas en una sola matriz
    X_encoded = np.hstack([X_cat_encoded, X[num_cols].values])
    # el objetivo también tiene que ser númerico, lo definimos así: abandono=0, matriculado=1, graduado=2
    label_map = {"abandono": 0, "matriculado": 1, "graduado": 2}
    label_names = ["abandono", "matriculado", "graduado"]
    y_num = y.map(label_map).values  # sustituye los textos por su número y lo convierte a array de numpy
    #80% train, 20% test
    #stratify para que las 3 clases queden bien repartidas
    #sin esto, "matriculado" (la clase con menos datos) puede quedar muy poco o nada representada en alguna partición
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_num, test_size=0.2, random_state=seed, stratify=y_num)

    #escalamos para la regresión logística(los árboles no necesitan escala)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Preprocesado terminado.")
    print(f"\nTenemos {len(X_train)} muestras en train(80%)")
    print(f"Tenemos {len(X_test)} muestras en test(20%)")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names, label_names



def validacion_cruzada(modelos, X_train_por_modelo, y_train):
    #KFold con 5 particiones: entrena con 4 y evalúa con 1, rotando hasta cubrir todo el train
    #usamos F1-macro como métrica principal porque da el mismo peso a las 3 clases
    #con accuracy el modelo podría ignorar "matriculado" y seguir sacando buena nota
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    resultados_cv = {}
    filas = []

    for nombre, modelo in modelos.items():
        scores = cross_validate(modelo, X_train_por_modelo[nombre], y_train, cv=cv,scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"}, n_jobs=-1)
        resultados_cv[nombre] = scores
        filas.append({
            "Modelo": nombre,
            "F1-macro": round(scores["test_f1_macro"].mean(), 4),
            "+-": round(scores["test_f1_macro"].std(), 4),
            "Accuracy": round(scores["test_accuracy"].mean(), 4),})

    print("\nVALIDACION CRUZADA KFOLD(k=5)")
    print(pd.DataFrame(filas).to_string(index=False))

    return resultados_cv


def evaluar_en_test(modelos, resultados_cv, X_train_por_modelo, X_test, X_test_scaled, y_train, y_test, label_names):
    # elegimos el mejor modelo según el F1-macro de CV
    mejor = None
    mejor_f1 = -1  # empezamos en -1 porque F1 siempre es >= 0, así cualquier modelo lo supera
    for nombre in resultados_cv:
        f1 = resultados_cv[nombre]["test_f1_macro"].mean()
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor = nombre
    print(f"\nMejor modelo (F1-macro CV): {mejor}")

    if mejor == "Regresion Logistica":
        X_test_final = X_test_scaled
    else:
        X_test_final = X_test

    modelo_final = modelos[mejor]
    modelo_final.fit(X_train_por_modelo[mejor], y_train)
    y_pred = modelo_final.predict(X_test_final)

    print(f"\nResultados en test:")
    print(f"  Accuracy:    {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1-macro:    {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"  F1-weighted: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    #la matriz de confusión muestra con qué clases acierta y con cuáles se confunde el modelo
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=label_names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de confusion ({mejor})")
    plt.tight_layout()
    plt.savefig("fig_confusion_matrix.png", dpi=150)
    plt.close()

    #tabla resumen con los 3 modelos para compararlos fácilmente
    filas = []
    for nombre, modelo in modelos.items():
        X_tr = X_train_por_modelo[nombre]

        if nombre == "Regresion Logistica":
            X_te = X_test_scaled
        else:
            X_te = X_test

        modelo.fit(X_tr, y_train)
        yp = modelo.predict(X_te)

        if nombre == mejor:
            etiqueta = nombre + " (mejor modelo)"
        else:
            etiqueta = nombre

        filas.append({
            "Modelo": etiqueta,
            "Accuracy": round(accuracy_score(y_test, yp), 4),
            "F1-macro": round(f1_score(y_test, yp, average="macro"), 4),
            "F1-weighted": round(f1_score(y_test, yp, average="weighted"), 4),
        })

    print("\nResumen final en test:")
    print(pd.DataFrame(filas).to_string(index=False))


def importancia_variables(modelos, X_train_por_modelo, y_train, feature_names):
    #con Random Forest podemos ver qué variables pesan más en la predicción
    rf = modelos["Random Forest"]
    rf.fit(X_train_por_modelo["Random Forest"], y_train)
    importancias = pd.Series(rf.feature_importances_, index=feature_names).nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    importancias.plot(kind="barh", ax=ax, color="#2ecc71", alpha=0.85)
    ax.set_title("Top 15 variables mas importantes (Random Forest)")
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    plt.savefig("fig_feature_importance.png", dpi=150)
    plt.close()

    print(f"\nTop 5 variables mas importantes:")
    print(importancias.tail(5).to_string())


def efecto_desbalanceo(X_train, X_test, y_train, y_test, label_names):
    #comparamos Random Forest con y sin class_weight="balanced" para ver cuánto importa el balanceo
    print("\n    Efecto de class_weight='balanced' (Random Forest)    ")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (etiqueta, cw) in zip(axes, [("Sin balanced", None), ("Con balanced", "balanced")]):
        rf_i = RandomForestClassifier(n_estimators=200, class_weight=cw, random_state=seed, n_jobs=-1)
        rf_i.fit(X_train, y_train)
        yp = rf_i.predict(X_test)
        ConfusionMatrixDisplay(confusion_matrix(y_test, yp), display_labels=label_names).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        f1 = f1_score(y_test, yp, average="macro")
        ax.set_title(f"RF {etiqueta}  |  F1-macro = {f1:.4f}")
        print(f"  {etiqueta}: F1-macro = {f1:.4f}")

    plt.suptitle("Impacto del balanceo de clases en Random Forest", fontweight="bold")
    plt.tight_layout()
    plt.savefig("fig_desbalanceo_comparacion.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    data = cargar_datos()
    print("\nDistribucion del objetivo:")
    print(data["objetivo"].value_counts().to_string())
    data = limpiar_datos(data)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names, label_names = preprocesar(data)

    modelos, X_train_por_modelo = buscar_mejores_modelos(X_train, X_train_scaled, y_train)

    resultados_cv = validacion_cruzada(modelos, X_train_por_modelo, y_train)

    evaluar_en_test(modelos, resultados_cv, X_train_por_modelo, X_test, X_test_scaled, y_train, y_test, label_names)

    importancia_variables(modelos, X_train_por_modelo, y_train, feature_names)

    efecto_desbalanceo(X_train, X_test, y_train, y_test, label_names)

    print("\nClasificacion completada.")
