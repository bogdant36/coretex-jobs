from typing import Optional
from pathlib import Path

import logging

from pandas import DataFrame
from coretex import CustomDataset

import pandas as pd


def loadData(dataset: CustomDataset) -> DataFrame:
    data: Optional[DataFrame] = None

    logging.info(">> [Activity Recognition] Downloading dataset...")
    dataset.download()

    logging.info(">> [Activity Recognition] Dataset downloaded. Loading data...")
    for i, sample in enumerate(dataset.samples):
        sample.unzip()
        if not i > 1:
            for path in Path(sample.path).iterdir():
                if not path.is_file():
                    continue

                if data is None:
                    data = pd.read_csv(path)
                    continue

                data = pd.concat([data, pd.read_csv(path)], ignore_index=True, sort = False)

    if data is None:
        raise RuntimeError(">> [Activity Recognition] Failed to load Activity Recognition data.")

    logging.info(f"[Activity Recognition] Data has been loaded. The shape of data is: {data.shape} .")

    return data
