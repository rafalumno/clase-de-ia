from pandas.core.frame import DataFrame
from pandas.core.series import Series
import pandas as pd


# Implementación de un clasificador Naive Bayes
class NaiveBayes:
    def __init__(self):
        # Variables para almacenar los datos de entrenamiento
        self.X = None
        self.Y = None
        self.df = None
        self.y_ocurrences = {}
        self.y_options_len = 0

    def fit(self, X: DataFrame, Y: Series, verbose=False):
        # Verifica que X y Y sean del tipo correcto
        if not isinstance(X, DataFrame):
            raise TypeError("X debe ser un DataFrame de pandas")
        if not isinstance(Y, Series):
            raise TypeError("Y debe ser una Series de pandas")

        # Guarda los datos y los concatena en un solo DataFrame
        self.X = X
        self.Y = Y
        self.df = pd.concat([X, Y], axis=1)

        # Obtiene las longitudes de X y Y
        x_length = len(X)
        y_length = len(Y)

        # Validaciones de tamaño por seguridad de datos
        if x_length < 1:
            raise ValueError("X no debe estar vacío")
        if y_length < 1:
            raise ValueError("Y no debe estar vacío")
        if x_length != y_length:
            raise ValueError("X e Y deben tener la misma cantidad de filas")

        # Cuenta cuántas clases únicas hay en Y
        self.y_options_len = len(Y.unique())
        # Cuenta las ocurrencias de cada clase en Y
        self.y_ocurrences = {}
        for label in Y.unique():
            self.y_ocurrences[label] = int(Y[Y == label].count())

        # Imprime las ocurrencias de cada clase si verbose es True
        if verbose:
            print(self.y_ocurrences)

    def predict(self, new_data: list, verbose=False):
        # Verifica que datos sean array
        if not isinstance(new_data, list):
            raise TypeError("new_data debe ser una lista")
        # Verifica que los datos sean del tipo aceptado
        for value in new_data:
            if not isinstance(value, (int, float, str)):
                raise TypeError("Los valores en new_data deben ser int, float o str")
        # Verifica que la cantidad de datos coincida con las columnas de X
        if len(new_data) != len(self.X.columns):
            raise ValueError("Cantidad de valores no coincide con contenido en X")

        # Calcula la probabilidad de cada clase dada la nueva información
        probabilities = {}
        # Itera sobre la cantidad de clases en Y
        for y_label, y_count in self.y_ocurrences.items():
            # Inicializa la probabilidad para la clase actual y arma el DataFrame
            prob = 1
            df_per_y = self.df[self.df[self.Y.name] == y_label]

            # Itera sobre cada valor en new_data
            for i, x_value in enumerate(new_data):
                # Guarda el nombre de la columna
                x_label = self.X.columns[i]
                # Cuenta cuántas veces aparece el valor de new_data en la columna actual
                series_per_x = df_per_y[x_label]
                x_count = int(series_per_x[series_per_x == x_value].count())
                # Aplica suavizado de Laplace y lo imprime si verbose es True
                if verbose:
                    print(f"({x_count} + 1) / ({y_count} + {self.y_options_len})")
                prob *= (x_count + 1) / (y_count + self.y_options_len)

            # Almacena la probabilidad calculada para la clase actual
            probabilities[y_label] = prob
            # Imprime la probabilidad por clase si verbose es True
            if verbose:
                print(f"P({y_label}|X) = {prob}")
        # Devuelve la clase con mayor probabilidad
        predicted_label = max(probabilities, key=probabilities.get)
        return predicted_label


# Ejemplo 1: Clasificación de clases escolares
df = pd.read_csv("clases.csv")
x = df[["horas", "intensidad"]]  # Variables predictoras
y = df["categoria"]  # Variable objetivo

model = NaiveBayes()  # Instancia del modelo
model.fit(x, y)  # Entrenamiento
prediccion = model.predict([4, 4])  # Predicción para clase no existente

print(f"La categoría predicha es: {prediccion}")  # Imprime la predicción

# Ejemplo 2: Clasificación de clases por calificaciones
df = pd.read_csv("calificaciones.csv")
x = df[["grade1", "grade2", "grade3", "grade4"]]  # Variables predictoras
y = df["class_name"]  # Variable objetivo

model = NaiveBayes()  # Instancia del modelo
model.fit(x, y)  # Entrenamiento
prediccion = model.predict([10.0, 10.0, 10.0, 10.0])  # Predicción de calificaciones

print(f"La clase predicha es: {prediccion}")  # Imprime la predicción

# Ejemplo 3: Predicción de mes de nacimiento por nombre
df = pd.read_csv("nombres.csv")
x = df[["name"]]  # Variable predictora
y = df["month"]  # Variable objetivo

model = NaiveBayes()  # Instancia del modelo
model.fit(x, y)  # Entrenamiento
prediccion = model.predict(["Jessica"])  # Predicción para un nombre

print(f"El mes predicho es: {prediccion}")  # Imprime la predicción
