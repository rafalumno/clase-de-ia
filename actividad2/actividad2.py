from pandas.core.frame import DataFrame
from pandas.core.series import Series
import pandas as pd


# Implementation of a Naive Bayes classifier
class NaiveBayes:
    def __init__(self):
        # Variables to store training data
        self.df = None
        self.x_column_names = []
        self.y_column_name = ""
        self.y_ocurrences = {}
        self.y_options_len = 0

    def fit(self, X: DataFrame, Y: Series, verbose=False):
        # Check that X and Y are of the correct type
        if not isinstance(X, DataFrame):
            raise TypeError("X debe ser un DataFrame de pandas")
        if not isinstance(Y, Series):
            raise TypeError("Y debe ser una Series de pandas")

        # Concatenate X and Y into a single DataFrame
        self.df = pd.concat([X, Y], axis=1)

        # Store column names
        self.x_column_names = X.columns.tolist()
        self.y_column_name = Y.name

        # Get the lengths of X and Y
        x_length = len(X)
        y_length = len(Y)

        # Size validations for data safety
        if x_length < 1:
            raise ValueError("X no debe estar vacío")
        if y_length < 1:
            raise ValueError("Y no debe estar vacío")
        if x_length != y_length:
            raise ValueError("X e Y deben tener la misma cantidad de filas")

        # Count how many unique classes are in Y
        self.y_options_len = len(Y.unique())
        # Count the occurrences of each class in Y
        self.y_ocurrences = {}
        for label in Y.unique():
            self.y_ocurrences[label] = int(Y[Y == label].count())

        # Print the occurrences of each class if verbose is True
        if verbose:
            print(self.y_ocurrences)

    def predict(self, new_data: list, verbose=False):
        # Check that new_data is a list
        if not isinstance(new_data, list):
            raise TypeError("new_data debe ser una lista")
        # Check that the data are of accepted types
        for value in new_data:
            if not isinstance(value, (int, float, str)):
                raise TypeError("Los valores en new_data deben ser int, float o str")
        # Check that the number of values matches the columns in X
        if len(new_data) != len(self.x_column_names):
            raise ValueError("Cantidad de valores no coincide con contenido en X")

        # Dictionary to store probabilities per class
        probabilities = {}
        # Iterate over the number of classes in Y
        for y_label, y_count in self.y_ocurrences.items():
            # Initialize the probability for the current class and build the DataFrame
            prob = 1
            df_per_y = self.df[self.df[self.y_column_name] == y_label]

            # Iterate over each value in new_data
            for i, x_value in enumerate(new_data):
                # Store the column name
                x_label = self.x_column_names[i]
                # Count how many times the value of new_data appears in the current column
                series_per_x = df_per_y[x_label]
                x_count = int(series_per_x[series_per_x == x_value].count())
                # Apply Laplace smoothing and print if verbose is True
                if verbose:
                    print(f"({x_count} + 1) / ({y_count} + {self.y_options_len})")
                prob *= (x_count + 1) / (y_count + self.y_options_len)

            # Store the calculated probability for the current class
            probabilities[y_label] = prob
            # Print the probability per class if verbose is True
            if verbose:
                print(f"P({y_label}|X) = {prob}")
        # Return the class with the highest probability
        predicted_label = max(probabilities, key=probabilities.get)
        return predicted_label


# Check if the script is being run directly
if __name__ == "__main__":
    # Model instance
    model = NaiveBayes()

    # Example 1: School class classification
    df = pd.read_csv("clases.csv")
    x = df[["horas", "intensidad"]]  # Predictor variables
    y = df["categoria"]  # Target variable

    model.fit(x, y)  # Training
    prediccion = model.predict([4, 4])  # Prediction for a non-existent class

    print(f"La categoría predicha es: {prediccion}")  # Print the prediction

    # Example 2: Class classification by grades
    df = pd.read_csv("calificaciones.csv")
    x = df[["grade1", "grade2", "grade3", "grade4"]]  # Predictor variables
    y = df["class_name"]  # Target variable

    model.fit(x, y)  # Training
    prediccion = model.predict([10.0, 10.0, 10.0, 10.0])  # Grade prediction

    print(f"La clase predicha es: {prediccion}")  # Print the prediction

    # Example 3: Predicting birth month by name
    df = pd.read_csv("nombres.csv")
    x = df[["name"]]  # Predictor variable
    y = df["month"]  # Target variable

    model.fit(x, y)  # Training
    prediccion = model.predict(["Jessica"])  # Prediction for a name

    print(f"El mes predicho es: {prediccion}")  # Print the prediction
