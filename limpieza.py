import pandas as pd


def cargar_datos():
    # cargamos el CSV con separador ; y mostramos cuántos estudiantes y variables tiene el dataset
    data = pd.read_csv("rendimiento_estudiantes.csv", sep=";")
    print(f"En el dataset tenemos {data.shape[0]} estudiantes y {data.shape[1]} variables")
    return data


def limpiar_base(data):
    # quitamos los 3 estudiantes con edad > 60( outliers claros, la mediana es de 20 años)
    data = data[data["edad_al_matricularse"] <= 60].copy()

    # eliminamos las variables macroeconómicas(iguales para todos los estudiantes del mismo periodo)
    data = data.drop(columns=["tasa_desempleo", "tasa_inflacion", "pib"])

    # tasa de aprobacion del 1er semestre: cuantas asignaturas aprueba de las que se presenta
    data["tasa_aprobacion_1sem"] = (
        data["asignaturas_1sem_aprobadas"] / data["asignaturas_1sem_evaluadas"].replace(0, 1)
    )

    return data
