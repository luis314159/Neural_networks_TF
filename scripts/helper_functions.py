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