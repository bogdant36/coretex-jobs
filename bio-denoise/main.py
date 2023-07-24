from pathlib import Path
from zipfile import ZipFile

from coretex import CustomDataset, CustomSample, Experiment, folder_manager
from coretex.project import initializeProject
from coretex.bioinformatics import qiime2 as ctx_qiime2


def dada2DenoiseSingleSample(sample: CustomSample, denoiseAlgorithm: str, trimLeft: int, truncLen: int, outputDir: Path) -> Path:
    samplePath = Path(sample.path)
    demuxPath = samplePath / "demux.qza"

    representativeSequencesPath = outputDir / "rep-seqs.qza"
    tablePath = outputDir / "table.qza"
    denoisingStatsPath = outputDir / "stats.qza"

    if denoiseAlgorithm == "DADA2":
        ctx_qiime2.dada2DenoiseSingle(
            str(demuxPath),
            trimLeft,
            truncLen,
            str(representativeSequencesPath),
            str(tablePath),
            str(denoisingStatsPath)
        )
    else:
        raise ValueError(f">> [Microbiome] {denoiseAlgorithm} not supported")

    denoiseOutput = outputDir / "denoise-output.zip"

    with ZipFile(denoiseOutput, "w") as denoiseFile:
        denoiseFile.write(representativeSequencesPath, "rep-seqs.qza")
        denoiseFile.write(tablePath, "table.qza")
        denoiseFile.write(denoisingStatsPath, "stats.qza")

    return denoiseOutput


def metadataTabulateSample(sample: CustomSample, outputDir: Path) -> Path:
    denoisingStatsPath = Path(sample.path) / "stats.qza"
    visualizationPath = outputDir / "stats.qzv"

    ctx_qiime2.metadataTabulate(str(denoisingStatsPath), str(visualizationPath))
    return visualizationPath


def featureTableSummarizeSample(sample: CustomSample, metadataPath: Path, outputDir: Path) -> Path:
    tablePath = Path(sample.path) / "table.qza"
    visualizationPath = outputDir / "table.qzv"

    ctx_qiime2.featureTableSummarize(str(tablePath), str(visualizationPath), str(metadataPath))
    return visualizationPath


def featureTableTabulateSeqsSample(sample: CustomSample, outputDir: Path) -> Path:
    inputPath = Path(sample.path) / "rep-seqs.qza"
    visualizationPath = outputDir / "rep-seqs.qzv"

    ctx_qiime2.featureTableTabulateSeqs(str(inputPath), str(visualizationPath))
    return visualizationPath


def processSample(
    index: int,
    sample: CustomSample,
    importedSample: CustomSample,
    experiment: Experiment,
    outputDataset: CustomDataset,
    outputDir: Path
):

    sampleOutputDir = outputDir / str(sample.id)
    sampleOutputDir.mkdir()

    # First step:
    # Denoise the demultiplexed sample generated by the previous step in the pipeline
    denoiseOutput = dada2DenoiseSingleSample(
        sample,
        experiment.parameters["algorithm"],
        experiment.parameters["trimLeft"],
        experiment.parameters["truncLen"],
        sampleOutputDir
    )

    denoisedSample = ctx_qiime2.createSample(f"{index}-denoise", outputDataset.id, denoiseOutput, experiment, "Step 2: Denoising")

    # Second step:
    # Generate visualization artifacts for the denoised data
    denoisedSample.download()
    denoisedSample.unzip()

    visualizationPath = metadataTabulateSample(denoisedSample, sampleOutputDir)
    ctx_qiime2.createSample(f"{index}-metadata-tabulate", outputDataset.id, visualizationPath, experiment, "Step 2: Denoising")

    # Third step:
    # Summarize how many sequences are associated with each sample and with each feature,
    # histograms of those distributions, and some related summary statistics
    metadataPath = Path(importedSample.path) / experiment.parameters["barcodesFileName"]
    featureTableSummaryPath = featureTableSummarizeSample(denoisedSample, metadataPath, sampleOutputDir)

    ctx_qiime2.createSample(f"{index}-feature-table-summarize", outputDataset.id, featureTableSummaryPath, experiment, "Step 2: Denoising")

    # Fourth step:
    # Provide a mapping of feature IDs to sequences,
    # and provide links to easily BLAST each sequence against the NCBI nt database
    featureTableMapPath = featureTableTabulateSeqsSample(denoisedSample, sampleOutputDir)
    ctx_qiime2.createSample(f"{index}-feature-table-tabulate-seqs", outputDataset.id, featureTableMapPath, experiment, "Step 2: Denoising")


def main(experiment: Experiment[CustomDataset]):
    experiment.dataset.download()

    demuxSamples = ctx_qiime2.getDemuxSamples(experiment.dataset)
    if len(demuxSamples) == 0:
        raise ValueError(">> [Workspace] Dataset has 0 demultiplexed samples")

    outputDir = folder_manager.createTempFolder("qiime_output")
    outputDataset = CustomDataset.createDataset(
        f"{experiment.id} - Step 2: Denoise",
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Workspace] Failed to create output dataset")

    for sample in demuxSamples:
        sample.unzip()

        index = ctx_qiime2.sampleNumber(sample)

        importedSample = experiment.dataset.getSample(f"{index}-import")
        if importedSample is None:
            raise ValueError(f">> [Workspace] Imported sample not found")

        importedSample.unzip()

        processSample(index, sample, importedSample, experiment, outputDataset, outputDir)


if __name__ == "__main__":
    initializeProject(main)
