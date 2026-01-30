import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("world_happiness_combined.csv", sep=";").dropna()

float_cols = [
    "Happiness score",
    "GDP per capita",
    "Social support",
    "Freedom to make life choices",
    "Generosity",
    "Perceptions of corruption",
]

for col in float_cols:
    df[col] = df[col].replace(",", ".", regex=True).astype(float)

latin_america = df[df["Regional indicator"] == "Latin America and Caribbean"]
latin_without_mexico = latin_america[latin_america["Country"] != "Mexico"]
mexico = df[df["Country"] == "Mexico"]

# 1. Listado de países más ricos de América Latina y donde se encuentra México.
gdp = (
    latin_america.groupby("Country")["GDP per capita"]
    .mean()
    .sort_values(ascending=True)
)
gdp.plot(
    kind="bar",
    title="Países más ricos de América Latina",
    color=["gray" if country != "Mexico" else "red" for country in gdp.index],
)
plt.ylabel("PIB per cápita")
plt.show()
# R: México se encuentra en la posición 9 dentro de los países más ricos de América Latina.

# 2. ¿Cuál es la relación entre Generosidad y Felicidad en México a través de los años?
mexico.plot(
    kind="line",
    x="Year",
    y=["Happiness score", "Generosity"],
    title="Generosidad vs Índice de Felicidad en México a través de los años",
)
plt.xlabel("Año")
plt.ylabel("Puntaje")
plt.legend(["Índice de Felicidad", "Generosidad"])
plt.show()
# R: En México, la generosidad no afecta a la felicidad porque la generosidad se mantiene estáble mientras que la felicidad varía.

# 3. ¿Cuál es el factor que más contribuye a la felicidad en México?
sns.heatmap(
    mexico.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
)
plt.title("Correlación de factores de felicidad en México")
plt.show()
# R: El factor que más contribuye a la felicidad en México es el PIB per cápita (GDP per capita).

# 4. Listado de países más corruptos de América Latina y donde se encuentra México.
corruption = (
    latin_america.groupby("Country")["Perceptions of corruption"]
    .mean()
    .sort_values(ascending=True)
)
corruption.plot(
    kind="bar",
    title="Países más corruptos de América Latina",
    color=["gray" if country != "Mexico" else "red" for country in corruption.index],
)
plt.ylabel("Percepción de corrupción")
plt.show()
# R: México se encuentra en la posición 5 dentro de los países más corruptos de América Latina.

# 5. ¿Ha aumentado o disminuido la felicidad en México a través de los años?
mexico.plot(
    kind="line",
    x="Year",
    y="Happiness score",
    title="Índice de felicidad de México a través de los años",
    xticks=df["Year"].unique(),
    color="red",
)
plt.xlabel("Año")
plt.ylabel("Índice de felicidad")
plt.show()
# R: La felicidad en México ha disminuido a través de los años. Comenzando en 2015 en su punto más alto y con descensos notables en 2018 y 2022 con una mejora en 2023 y 2024.

# 6. Comparación de la felicidad de México con el promedio regional.
latin_without_mexico.groupby("Year")["Happiness score"].mean().plot(
    kind="line",
    title="Índice de felicidad: México vs Promedio de América Latina",
    xticks=df["Year"].unique(),
)
mexico.groupby("Year")["Happiness score"].mean().plot(
    kind="line",
    color="red",
)
plt.legend(["Promedio América Latina", "México"])
plt.xlabel("Año")
plt.ylabel("Índice de felicidad")
plt.title("Índice de felicidad: México vs Promedio de América Latina")
plt.show()
# R: La felicidad en México es superior al promedio de América Latina en todos los años, aunque incluye más variabilidad.

# 7. ¿El PIB per cápita afecta el soporte social en México?
sns.regplot(
    data=mexico,
    x="GDP per capita",
    y="Social support",
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"},
)
plt.title("Soporte social vs PIB per cápita en México")
plt.show()
# R: En México, cuando el PIB per cápita aumenta, el soporte social parece decreser.

# 8. ¿Cuál ha sido el factor con más variabilidad en México a través de los años?
mexico[float_cols].std().sort_values(ascending=False).plot(
    kind="barh",
    title="Variabilidad de los factores de felicidad en México a través de los años",
    color="orange",
)
plt.show()
# R: El factor con más variabilidad en México ha sido el PIB per cápita (GDP per capita).

# 9. ¿La libertad se traduce en mayor felicidad en México?
sns.regplot(
    data=mexico,
    x="Freedom to make life choices",
    y="Happiness score",
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"},
)
plt.title("Índice de felicidad vs Libertad para tomar decisiones en México")
plt.show()
# R: En México, una mayor libertad para tomar decisiones no significa mayor felicidad.

# 10. ¿El soporte social se debe al nivel de generosidad en México?
mexico.plot(
    kind="bar",
    x="Year",
    y=["Social support", "Generosity"],
    title="Soporte social vs Generosidad en México a través de los años",
)
plt.ylabel("Puntaje")
plt.show()
# R: No hay evidencia suficiente para afirmar que el soporte social se deba al nivel de generosidad en México, ya que con diferentes niveles de generosidad, el soporte social no obtiene una tendencia positiva o negativa clara.
