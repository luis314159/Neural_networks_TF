def load_iris_DataSet():
    from sklearn.datasets import load_iris
    import pandas as pd

    # Load the iris dataset from sklearn
    iris = load_iris()

    # Convert the iris dataset to a pandas dataframe
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Add the target variable to the dataframe
    df['target'] = iris.target
    return df

def load_iris_DataSet():
    """
    Carga el conjunto de datos Iris desde la biblioteca scikit-learn y lo convierte en un DataFrame de pandas.

    Retorna:
    - df: DataFrame de pandas con las características del conjunto de datos Iris y la variable objetivo 'target'.
    """
    
    # Importar las bibliotecas necesarias
    from sklearn.datasets import load_iris
    import pandas as pd

    # Cargar el conjunto de datos Iris desde scikit-learn
    iris = load_iris()

    # Convertir el conjunto de datos Iris a un DataFrame de pandas
    # Usar los nombres de características de Iris como nombres de columnas
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Agregar la variable objetivo (target) al DataFrame
    # Esta variable indica la especie de cada flor: 0 = setosa, 1 = versicolor, 2 = virginica
    df['target'] = iris.target
    
    # Retornar el DataFrame
    return df



def data_graph_2D(df_valores, df_clases):
    """
    Grafica los datos en 2D utilizando las coordenadas de df_valores y la clase de df_clases.
    
    Parámetros:
    - df_valores: DataFrame con columnas para las coordenadas en el plano 2D.
    - df_clases: DataFrame con una columna que representa la clase de cada punto en df_valores.
    
    Retorna:
    None. Muestra una gráfica 2D.
    """
    
    # Importaciones necesarias
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Asegurarse de que ambos dataframes tengan la misma longitud
    if len(df_valores) != len(df_clases):
        print("Los dataframes no tienen la misma longitud.")
        return
    
    # Combinar ambos dataframes en uno solo para facilitar la graficación
    df_combinado = pd.concat([df_valores, df_clases], axis=1)
    
    # Extraer los nombres de las columnas
    x_col, y_col, clase_col = df_combinado.columns
    
    # Iterar sobre las clases únicas y graficar cada conjunto de puntos en el plano 2D
    for etiqueta_clase in df_combinado[clase_col].unique():
        plt.scatter(df_combinado[df_combinado[clase_col] == etiqueta_clase][x_col], 
                    df_combinado[df_combinado[clase_col] == etiqueta_clase][y_col], 
                    label=etiqueta_clase)
    
    # Configurar la leyenda, título y etiquetas de los ejes
    plt.legend()
    plt.title("Gráfico de dispersión de valores por clase")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    
    # Mostrar la gráfica
    plt.show()


def graficar_dataframes_3D(df_valores, df_clases):
    """
    Grafica los datos en 3D tomando las coordenadas de df_valores y la clase de df_clases.
    
    Parámetros:
    - df_valores: DataFrame con columnas 'x', 'y', 'z' representando las coordenadas en 3D.
    - df_clases: DataFrame con una columna 'Clase' representando la clase de cada punto en df_valores.
    
    Retorna:
    None. Muestra una gráfica 3D.
    """
    
    # Importaciones necesarias
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Asegurarse de que ambos dataframes tengan la misma longitud
    if len(df_valores) != len(df_clases):
        print("Los dataframes no tienen la misma longitud.")
        return
    
    # Combinar ambos dataframes en uno solo para facilitar la graficación
    df_combinado = pd.concat([df_valores, df_clases], axis=1)
    
    # Extraer los nombres de las columnas
    x_col, y_col, z_col, clase_col = df_combinado.columns
    
    # Inicializar la figura y el eje 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Iterar sobre las clases únicas y graficar cada conjunto de puntos en el espacio 3D
    for etiqueta_clase in df_combinado[clase_col].unique():
        ax.scatter(df_combinado[df_combinado[clase_col] == etiqueta_clase][x_col], 
                   df_combinado[df_combinado[clase_col] == etiqueta_clase][y_col],
                   df_combinado[df_combinado[clase_col] == etiqueta_clase][z_col], 
                   label=etiqueta_clase)
    
    # Configurar la leyenda, título y etiquetas de los ejes
    ax.legend()
    ax.set_title("Gráfico de dispersión 3D de valores por clase")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    
    # Mostrar la gráfica
    plt.show()

def plot_decision_regions(data, labels, classifier, xlabel=None, ylabel=None, legend_loc=None, resolution=0.02):
    """
    Visualiza las regiones de decisión de un clasificador en 2D.
    
    Parámetros:
    - data: Datos de entrada (features).
    - labels: Etiquetas o clases de los datos.
    - classifier: Clasificador entrenado.
    - xlabel: Etiqueta para el eje X.
    - ylabel: Etiqueta para el eje Y.
    - legend_loc: Ubicación de la leyenda en la gráfica.
    - resolution: Resolución de la malla de decisión.
    
    Retorna:
    None. Muestra una gráfica.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    data, labels = np.array(data), np.array(labels)

    markers = ('s', 'x', 'o', '^', 'v', '*', 'p', 'D', 'H', '<', '>', '1', '2', '3', '4')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'magenta', 'yellow', 'white', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral')
    cmap = ListedColormap(colors[:len(np.unique(labels))])

    x1_min, x1_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    x2_min, x2_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    input = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = classifier.predict(input)
    
    print(Z.ndim, Z.shape[1] )
    if Z.ndim > 1 and Z.shape[1] > 1:
        Z = np.argmax(Z, axis=1)
    else:
        Z = np.squeeze(Z)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    for idx, cl in enumerate(np.unique(labels)):
        plt.scatter(x=data[labels == cl, 0], y=data[labels == cl, 1],
                    alpha=0.8, c=colors[idx % len(colors)], marker=markers[idx % len(markers)], 
                    label=cl, edgecolor='black')

    if xlabel:
        plt.xlabel(xlabel) 
    if ylabel:
        plt.ylabel(ylabel)
    if legend_loc:
        plt.legend(loc=legend_loc)

    plt.show()


def plot_decision_plane_3D_interactive(data, labels, classifier, resolution=0.02, opacity = 0.08):
    # Convertir a numpy array si es necesario
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    if isinstance(data, pd.DataFrame):
      
        data = data.values

    # Extraer límites para cada dimensión
    x1_min, x1_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    x2_min, x2_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    x3_min, x3_max = data[:, 2].min() - 1, data[:, 2].max() + 1

    # Generar grillas para dos de los tres ejes (x y y)
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # Crear trazas para datos
    traces = []
    unique_labels = np.unique(labels)
    for idx, label in enumerate(unique_labels):
        traces.append(go.Scatter3d(
            x=data[labels == label, 0],
            y=data[labels == label, 1],
            z=data[labels == label, 2],
            mode='markers',
            marker=dict(size=5, opacity=0.8),
            name=f"Class {label}"
        ))

    # Trazar la superficie de decisión en varios niveles de z
    z_values = np.arange(x3_min, x3_max, resolution)
    for z in z_values:
        Z = classifier.predict(np.c_[xx1.ravel(), xx2.ravel(), np.full(xx1.ravel().shape, z)])
        Z = Z.reshape(xx1.shape)
        surface = go.Surface(x=xx1, y=xx2, z=z * np.ones_like(xx1), surfacecolor=Z, colorscale='Viridis', opacity=opacity, showscale=False)
        traces.append(surface)

    # Crear la figura
    layout = go.Layout(title="3D Decision Plane")
    fig = go.Figure(data=traces, layout=layout)
    
    # Mostrar la figura
    fig.show()


def plot_decision_boundary_3D(data, labels, classifier, resolution=0.02, opacity = 0.03):
    # Convertir a numpy array si es necesario
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Extraer límites para cada dimensión
    x1_min, x1_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    x2_min, x2_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    x3_min, x3_max = data[:, 2].min() - 1, data[:, 2].max() + 1

    # Generar grillas para los tres ejes
    xx1, xx2, xx3 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                np.arange(x2_min, x2_max, resolution),
                                np.arange(x3_min, x3_max, resolution))

    # Predicciones para cada combinación de puntos en la grilla
    Z = classifier.predict(np.c_[xx1.ravel(), xx2.ravel(), xx3.ravel()])
    Z = Z.reshape(xx1.shape)

    # Vamos a buscar puntos cerca de los límites de decisión
    # Para simplificar, identificamos puntos donde el vecino no tiene el mismo valor
    boundary_points = np.where(
        (np.roll(Z, shift=-1, axis=0) != Z) |
        (np.roll(Z, shift=-1, axis=1) != Z) |
        (np.roll(Z, shift=-1, axis=2) != Z)
    )

    # Crear trazas para datos
    traces = []
    unique_labels = np.unique(labels)
    for idx, label in enumerate(unique_labels):
        traces.append(go.Scatter3d(
            x=data[labels == label, 0],
            y=data[labels == label, 1],
            z=data[labels == label, 2],
            mode='markers',
            marker=dict(size=5, opacity=opacity),
            name=f"Class {label}"
        ))

    # Trazar la superficie de decisión
    surface = go.Mesh3d(x=xx1[boundary_points],
                        y=xx2[boundary_points],
                        z=xx3[boundary_points],
                        alphahull=5, opacity=0.4, color='yellow')
    traces.append(surface)

    # Crear la figura
    layout = go.Layout(title="3D Decision Boundary")
    fig = go.Figure(data=traces, layout=layout)

    # Mostrar la figura
    fig.show()


def plot_3D_interactive(data, labels):
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    """
    Grafica puntos de datos en 3D de manera interactiva utilizando plotly.
    
    Parámetros:
    - data: Un array o DataFrame con tres columnas, representando las coordenadas x, y, z de los puntos.
    - labels: Una lista o array con las etiquetas (clases) de cada punto en `data`.

    Retorna:
    None. Muestra un gráfico 3D interactivo.
    """
    
    # Convertir a numpy array si 'data' es un DataFrame de pandas
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Inicializar una lista para almacenar las trazas (series de datos) que se mostrarán en el gráfico
    traces = []
    
    # Obtener etiquetas únicas para poder colorear los puntos según su clase
    unique_labels = np.unique(labels)
    
    # Por cada etiqueta única, crea una traza (serie de datos) y añádela a la lista 'traces'
    for idx, label in enumerate(unique_labels):
        traces.append(go.Scatter3d(
            x=data[labels == label, 0],   # Coordenadas x de los puntos con la etiqueta actual
            y=data[labels == label, 1],   # Coordenadas y de los puntos con la etiqueta actual
            z=data[labels == label, 2],   # Coordenadas z de los puntos con la etiqueta actual
            mode='markers',               # Tipo de gráfico (en este caso, marcadores/puntos)
            marker=dict(size=5, opacity=0.8), # Estilo de los marcadores
            name=f"Class {label}"         # Nombre de la traza (etiqueta)
        ))

    # Definir la disposición (layout) del gráfico, como el título
    layout = go.Layout(title="3D Data Plot")
    
    # Crear la figura usando las trazas y el layout definidos
    fig = go.Figure(data=traces, layout=layout)
    
    # Mostrar la figura
    fig.show()