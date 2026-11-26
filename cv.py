import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

seed = 42  # fijamos la semilla para que los resultados sean siempre los mismos


def buscar_mejores_modelos(X_train, X_train_scaled, y_train):
    # mismas 5 particiones que en el resto del proyecto para poder comparar resultados
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    # un diccionario por modelo con: el objeto base, la rejilla de parámetros a probar y qué datos usa
    # la regresión logística usa datos escalados, los árboles no lo necesitan
    configuraciones = {
        "Regresion Logistica": {
            # C es la inversa de la regularización: C pequeño = más regularización, C grande = menos
            "modelo": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            "grid": {"C": [0.01, 0.1, 1, 10, 100]},
            "X": X_train_scaled,
        },
        "Random Forest": {
            "modelo": RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
            # probamos más/menos árboles, con o sin límite de profundidad y distintos tamaños mínimos de nodo
            "grid": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
            "X": X_train,
        },
        "Gradient Boosting": {
            "modelo": GradientBoostingClassifier(random_state=seed),
            # probamos número de árboles, profundidad y la tasa de aprendizaje (cuánto pesa cada árbol nuevo)
            "grid": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5],
            },
            "X": X_train,
        },
    }

    print("\nBUSQUEDA DE HIPERPARAMETROS (GridSearchCV, k=5, F1-macro)")
    modelos = {}
    X_train_por_modelo = {}

    # para cada modelo: lanzamos el grid search, imprimimos los mejores params y guardamos el modelo ganador
    for nombre, cfg in configuraciones.items():
        # GridSearchCV prueba todas las combinaciones de la rejilla con validación cruzada
        # scoring="f1_macro" para que evalúe igual las 3 clases aunque estén desbalanceadas
        gs = GridSearchCV(
            cfg["modelo"],
            cfg["grid"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,  # con refit=True el mejor modelo ya queda reentrenado con todo el train al terminar
        )
        gs.fit(cfg["X"], y_train)
        modelos[nombre] = gs.best_estimator_  # el modelo con los mejores hiperparámetros, ya entrenado
        X_train_por_modelo[nombre] = cfg["X"]
        print(f"  {nombre}:")
        print(f"    mejores params : {gs.best_params_}")
        print(f"    F1-macro CV    : {gs.best_score_:.4f}")

    return modelos, X_train_por_modelo


def buscar_mejores_modelos_regresion(X_train, X_train_scaled, y_train):
    # mismas 5 particiones que en el resto del proyecto para poder comparar resultados
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    # regresion lineal no tiene hiperparámetros que tunear, la añadimos directamente
    modelos = {"Regresion Lineal": LinearRegression()}
    X_train_por_modelo = {"Regresion Lineal": X_train_scaled}

    # un diccionario por modelo con: el objeto base, la rejilla de parámetros a probar y qué datos usa
    configuraciones = {
        "Ridge": {
            # alpha controla la regularización: valores grandes penalizan más los coeficientes grandes
            "modelo": Ridge(),
            "grid": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "X": X_train_scaled,
        },
        "Random Forest": {
            "modelo": RandomForestRegressor(random_state=seed, n_jobs=-1),
            # probamos más/menos árboles, con o sin límite de profundidad y distintos tamaños mínimos de nodo
            "grid": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
            "X": X_train,
        },
        "Gradient Boosting": {
            "modelo": GradientBoostingRegressor(random_state=seed),
            # probamos número de árboles, profundidad y la tasa de aprendizaje
            "grid": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5],
            },
            "X": X_train,
        },
    }

    print("\nBUSQUEDA DE HIPERPARAMETROS REGRESION (GridSearchCV, k=5, RMSE)")
    print("  Regresion Lineal: sin hiperparametros que tunear")

    # para cada modelo: lanzamos el grid search, imprimimos los mejores params y guardamos el modelo ganador
    for nombre, cfg in configuraciones.items():
        # scoring con neg_root_mean_squared_error porque sklearn minimiza, así el mayor valor = menor RMSE
        gs = GridSearchCV(
            cfg["modelo"],
            cfg["grid"],
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            refit=True,  # con refit=True el mejor modelo ya queda reentrenado con todo el train al terminar
        )
        gs.fit(cfg["X"], y_train)
        modelos[nombre] = gs.best_estimator_  # el modelo con los mejores hiperparámetros, ya entrenado
        X_train_por_modelo[nombre] = cfg["X"]
        print(f"  {nombre}:")
        print(f"    mejores params : {gs.best_params_}")
        print(f"    RMSE CV        : {-gs.best_score_:.4f}")

    return modelos, X_train_por_modelo
