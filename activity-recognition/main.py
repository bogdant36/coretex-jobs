import logging

from kerastuner.tuners import RandomSearch
from coretex import Experiment, CustomDataset
from coretex.folder_management import FolderManager
from coretex.project import initializeProject

import tensorflow as tf
import coremltools as ct
import tensorflowjs as tfjs

from src.load_data import loadData
from src.process_data import processData
from src.model import buildModel, uploadModel


def main(experiment: Experiment[CustomDataset]) -> None:
    modelDir = FolderManager.instance().createTempFolder("modelFolder")
    logging.info(f"[Activity Recognition] Fetching parameters from experiment.config file...")

    epochs = experiment.parameters["epochs"]
    maxTrials = experiment.parameters["maxTrials"]
    experimentsPerTrial = experiment.parameters["experimentsPerTrial"]
    numOfModels = experiment.parameters["numOfModels"]
    maxTrials = experiment.parameters["maxTrials"]
    validationSplit = experiment.parameters["validationSplit"]
    targetColumn = experiment.parameters["targetColumn"]

    data = loadData(experiment.dataset)
    dataLength = data.shape[0]

    logging.info(f"[Activity Recognition] Processing data for training.")
    xTrain, xTest, yTrain, yTest, labels = processData(data, targetColumn, validationSplit)

    tuner = RandomSearch(
        buildModel,
        objective = 'val_accuracy',
        max_trials = maxTrials,
        executions_per_trial = experimentsPerTrial,
        directory = 'project',
        project_name = 'Human_activity_recognition'
    )

    tuner.search_space_summary()
    tuner.search(xTrain, yTrain,
        epochs = epochs,
        validation_data = (xTest, yTest))

    model = tuner.get_best_models(num_models = numOfModels)[0] # returns the best model

    Callback = tf.keras.callbacks.EarlyStopping(monitor = 'accuracy', patience = 3)
    mo_fitt = model.fit(xTrain, yTrain, epochs = epochs, validation_data = (xTest, yTest), callbacks = Callback)

    accuracy = mo_fitt.history['accuracy'][-1]
    loss = mo_fitt.history['loss'][-1]
    validation_loss = mo_fitt.history['val_loss'][-1]
    validation_accuracy = mo_fitt.history['val_accuracy'][-1]

    logging.info(f"[Activity Recognition] Job finished. Logging metrics...")

    logging.info(f"[Activity Recognition] Accuracy: {accuracy}")
    logging.info(f"[Activity Recognition] Loss: {loss}")
    logging.info(f"[Activity Recognition] Validation loss: {validation_loss}")
    logging.info(f"[Activity Recognition] Validation accuracy: {validation_accuracy}")

    # Save the model in TensorFlow SavedModel format
    model.save(f"{modelDir}/tf_saved_model")

    # Save the model in TensorFlow.js format
    tfjs.converters.save_keras_model(model, f"{modelDir}/tfjs_model")

    # Save the model in CoreML format
    coreml_model = ct.converters.convert(model, source='tensorflow')
    coreml_model.save(f"{modelDir}/model.mlmodel")

    uploadModel(accuracy, labels, dataLength, experiment)


if __name__ == "__main__":
    initializeProject(main)
