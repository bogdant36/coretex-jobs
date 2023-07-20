from tensorflow import keras
from keras.layers import Dropout
from tensorflow.keras import layers

from coretex import Experiment, CustomDataset, Model
from coretex.folder_management import FolderManager


def buildModel(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 25)):
        model.add(layers.Dense(units = hp.Int('units' + str(i), min_value=32, max_value=512, step=32),
                               kernel_initializer= hp.Choice('initializer', ['uniform', 'normal']),
                               activation= hp.Choice('activation', ['relu', 'sigmoid', 'tanh'])))
    model.add(layers.Dense(9, kernel_initializer= hp.Choice('initializer', ['uniform', 'normal']), activation='softmax'))
    model.add(
            Dropout(0.2))
    model.compile(
        optimizer = 'adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


def uploadModel(accuracy: float, labels: list[int], dataLength: int, experiment: Experiment[CustomDataset]) -> None:
    modelPath = FolderManager.instance().getTempFolder("modelFolder")

    model = Model.createModel(experiment.name, experiment.id, accuracy, {})
    model.saveModelDescriptor(modelPath, {
        "project_task": experiment.spaceTask,
        "labels": labels,
        "modelName": model.name,
        "description": experiment.description,

        "input_description": """
            Input shape is []

            - 
            - 
        """,
        "input_shape": [dataLength, len(labels)],

        "output_description": """
            Output shape - []

            - 
            - 
        """,
        "output_shape": [dataLength, 1]
    })

    model.upload(modelPath)
