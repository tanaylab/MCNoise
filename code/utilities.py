"""
Holds several useful utility functions which are either shared between different objects or require the user to define general things before the pipeline.

- get_empty_droplets_total_genes_count - Collect and extract information from all the empty droplets across the different batches files.

- get_number_of_processes_to_use - Calculate the number of processes we can and need to use for a multiprocess operation.

- calculate_negative_loglikelihood - Helpful function to calculate a negative log-likelihood between predicted and observed values.

- remove_uncommon_genes_from_cells_metacells_empty_droplets_files - Slice the cells adata, metacell adata and empty droplets information to share the same genes set.

"""


import multiprocessing
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import scipy.special


def get_number_of_processes_to_use(
    number_of_requested_processes: int, number_of_tasks: int, logical_cpu=True
) -> int:
    """
    Find the number of processes to use in a multiprocess operation.
    This cannot be more than the number of processes the computer has (logical or not), or negative.
    Also, this should not be more than the number of tasks we need.

    :param number_of_requested_processes: The requested number of processes.
    :type number_of_requested_processes: int

    :param number_of_tasks: The number of tasks we want to multiprocess; this is the upper limit of the number of processes we need.
    :type number_of_tasks: int

    :param logical_cpu: Logical CPU is better used for our cases, which does a lot of I/O operation. Defaults to True, defaults to True.
    :type logical_cpu: bool, optional

    :return: The number of processes we should use to the tasks based on the given paramaters.
    :rtype: int
    """

    num_of_available_processes = psutil.cpu_count(logical=logical_cpu)

    # Cannot use more than what the user-defined, but also cannot use something less than 1.
    number_of_processes_to_use = max(
        min(number_of_requested_processes, num_of_available_processes), 1
    )

    return min(number_of_tasks, number_of_processes_to_use)


def calculate_negative_loglikelihood(observed: pd.Series, expected: pd.Series) -> float:
    """
    Calculate the negative log-likelihood of the observed vs. expected data in a Poisson distribution.

    :param observed: The observed values in the data.
    :type observed: pd.Series

    :param expected: The expected values in the data.
    :type expected: pd.Series

    :return: Negative log-likelihood value of the observed vs. expected data.
    :rtype: float
    """
    return -expected + np.log(expected) * observed - scipy.special.gammaln(observed)  # type: ignore


def remove_uncommon_genes_from_cells_metacells_empty_droplets_files(
    cells_adata: ad.AnnData,
    metacells_adata: ad.AnnData,
    batches_empty_droplets_dict: dict[str, pd.Series],
) -> tuple[ad.AnnData, ad.AnnData, dict[str, pd.Series]]:
    """
    Make sure that the cells anndata and the metacell anndata share the same genes with the empty droplets files.
    In many cases, this might not be true, and then we clip those objects without them.
    This is likely happening due to different versions of pipleines generated those files.


    :param cells_adata: The full cells adata
    :type cells_adata: ad.AnnData

    :param metacells_adata: The full metacell adata
    :type metacells_adata: ad.AnnData

    :param batches_empty_droplets_dict: Mapping between batch name to a series of umi count per gene in the empty droplets.
    :type batches_empty_droplets_dict: dict[str, pd.Series]

    :return: The cells adata, metacells adata and mapping between the batch name and empty droplet count of genes, now sharing the same gene set.
    :rtype: tuple[ad.AnnData, ad.AnnData, dict[str, pd.Series]]
    """

    common_genes = cells_adata.var.index & metacells_adata.var.index
    for batch in batches_empty_droplets_dict:
        common_genes = common_genes & batches_empty_droplets_dict[batch].index

    cells_adata = cells_adata[:, common_genes]
    metacells_adata = metacells_adata[:, common_genes]
    batches_empty_droplets_dict = {
        batch: batches_empty_droplets_dict[batch].loc[common_genes]
        for batch in batches_empty_droplets_dict
    }

    return cells_adata, metacells_adata, batches_empty_droplets_dict


def get_empty_droplets_total_genes_count(
    batches_to_file_path: dict[str, str],
    read_file_func: Callable[[str], ad.AnnData],
    max_umi_count_to_be_empty_droplet: int = 100,
    number_of_processes: int = 1,
) -> dict[str, pd.Series]:
    """
    Go over all the batches files and extract the empty droplets umi count per gene from each file.
    This can be done file by file or in a multiprocess fashion.

    :param batches_to_file_path: 
        Mapping between the batch name and the droplets source file. 
        This need to be an h5 file, mtx file, or some other representation that extracts Anndata object after using the `read_file_func` function
    :type batches_to_file_path: dict[str, str]

    :param read_file_func: A function that gets a file path of single-cell data and reads it as AnnData file with the droplet, genes, and umi information.
    :type read_file_func: Callable[[str], ad.AnnData]

    :param max_umi_count_to_be_empty_droplet: Droplets with fewer umi count will be considered as empty droplet for distribution calculation, defaults to 100.
    :type max_umi_count_to_be_empty_droplet: int, optional

    :param number_of_processes: Allow to run each batch file in a different process, defaults to 1.
    :type number_of_processes: int, optional

    :return: Mapping between each batch and its empty droplets umis count per gene.
    :rtype: dict[str, pd.Series]
    """
    number_of_processes = get_number_of_processes_to_use(
        number_of_requested_processes=number_of_processes,
        number_of_tasks=len(batches_to_file_path),
    )

    if number_of_processes > 1:
        with multiprocessing.Pool(number_of_processes) as multiprocess_pool:
            empty_droplets_distribution = multiprocess_pool.starmap(
                _get_empty_droplets_distribution_from_file,
                [
                    (
                        batch,
                        batches_to_file_path[batch],
                        read_file_func,
                        max_umi_count_to_be_empty_droplet,
                    )
                    for batch in batches_to_file_path
                ],
            )

    else:
        empty_droplets_distribution = [
            _get_empty_droplets_distribution_from_file(
                batch_name=batch,
                path_to_droplets_file=batches_to_file_path[batch],
                read_file_func=read_file_func,
                max_umi_count_to_be_empty_droplet=max_umi_count_to_be_empty_droplet,
            )
            for batch in batches_to_file_path
        ]

    batch_empty_droplets = {
        batch: empty_droplets_distribution[i]
        for i, batch in enumerate(batches_to_file_path)
    }
    return batch_empty_droplets


def _get_empty_droplets_distribution_from_file(
    batch_name: str,
    path_to_droplets_file: str,
    read_file_func: Callable[[str], ad.AnnData],
    max_umi_count_to_be_empty_droplet: int,
) -> pd.Series:
    """
    Read a droplet file and extract the empty droplets' umi count. 
    All the droplets with at most max_umi_count_to_be_empty_droplet umis are considered empty and will be aggregated together to get the umi count per gene. 
    This later can be used to calculate the fraction and frequency of gene distribution in the ambient noise.

    :param batch_name: The name of the batch, used for logging.
    :type batch_name: str

    :param path_to_droplets_file: A valid path to the droplets file, of type h5, mtx, etc and should match the reading function `read_file_func`.
    :type path_to_droplets_file: str

    :param read_file_func: 
        A function that gets a file path of single-cell data and reads it as AnnData file with the droplet, genes, and umi information.
        Examples can be sc.read_h5, sc.read_mtx, etc.
    :type read_file_func: Callable[[str], ad.AnnData]

    :param max_umi_count_to_be_empty_droplet: Droplets with fewer umi count will be considered as empty droplet for distribution calculation.
    :type max_umi_count_to_be_empty_droplet: int

    :return: Return the total umis count per gene in all the empty droplets.
    :rtype: pd.Series
    """
    
    batch_dropelts_adata = read_file_func(path_to_droplets_file)

    # Find all the droplets which are considered as empty base on the threshold.
    droplet_count = batch_dropelts_adata.X.sum(axis=1)  
    empty_droplets = batch_dropelts_adata.X[  
        np.where(
            (droplet_count > 0) & (droplet_count < max_umi_count_to_be_empty_droplet)
        )[0],
        :,
    ]

    # TODO: logging and not print
    print(  # Do not print from a package. Log, if you have to.
        "%s had %d(%1.f%%) out of %d empty droplets"
        % (
            batch_name,
            empty_droplets.shape[0],
            100 * empty_droplets.shape[0] / len(droplet_count),
            len(droplet_count),
        )
    )

    # Aggregate the umis from all the empty droplets.
    empty_droplets_series = pd.Series(
        np.array(empty_droplets.sum(axis=0)).reshape(-1),
        index=batch_dropelts_adata.var.index,
    )
    empty_droplets_series = empty_droplets_series.groupby(level=0).sum()
    return empty_droplets_series.squeeze()
