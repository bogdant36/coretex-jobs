from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import pandas as pd


def processData(data: DataFrame, targetColumn: str, validationSplit: int = 0.2) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, list[int]]:
    X = data.drop([targetColumn, 'timestamp'], axis=1)
    y = pd.DataFrame(data[targetColumn])

    labels = y[targetColumn].unique()
    labels.sort()
    labels = labels.tolist()

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = validationSplit, random_state = 42)

    scaling_data = MinMaxScaler()
    xTrain = scaling_data.fit_transform(xTrain)
    xTest = scaling_data.transform(xTest)

    return xTrain, xTest, yTrain, yTest, labels
