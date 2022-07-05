"""
Holds several usefull utilities functions which are either shared between different objects or require the user to define general things before the pipeline.
This includes:

- get_empty_droplets_distribution - Collect and extract information from all the empty droplets across the different batches files.
- get_number_of_valid_processes - Validate we don't ask too much processes during a multiprocess operation.
- calculate_negative_loglikelihood - Helpful function to calculate a negative log likelihood between predicted and observed values.
"""


import functools
import multiprocessing
import os
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import scipy.special


def get_empty_droplets_distribution(
    batches_to_file_path: dict[str, str],
    read_file_func: Callable[[str], ad.AnnData],
    max_umi_count_to_be_empty_droplet=100,
    number_of_processes=np.inf,
) -> dict[str, pd.DataFrame]:
    """Go over all the batches files and extract the empty droplets distribution from each file.
    This can be done file by file or in a multiprocess fashion.

    Args:
        batches_to_file_path (dict[str,str]): Mapping between the batch name and the droplets source file. From this file we will extract the empty droplets.

        read_file_func (Callable[[str], ad.AnnData]): A function that get  a file path of single cell data and read it as AnnData file with the droplet, genes and umi information.
        Examples can be sc.read_h5, sc.read_mtx, and etc.

        max_umi_count_to_be_empty_droplet (int, optional): Droplets with fewer umi count will be considered as empty droplet for distribution calculation. Defaults to 100.

        number_of_processes (_type_, optional): Allow to run each batch file in a different process. Defaults to np.inf.

    Returns:
        dict[str, pd.DataFrame]: Mapping between each batch and it's empty droplets umis distributions. This distribution is still in count format so integers.
    """
    number_of_processes = get_number_of_valid_processes(number_of_processes)

    if number_of_processes > 1:
        with multiprocessing.Pool(number_of_processes) as multiprocess_pool:
            empty_droplets_distribution = multiprocess_pool.map(
                functools.partial(
                    _get_empty_droplets_distribution_from_file,
                    read_file_func=read_file_func,
                    max_empty_droplet_umis=max_umi_count_to_be_empty_droplet,
                ),
                [
                    (batches_to_file_path[batch], batch)
                    for batch in batches_to_file_path
                ],
            )

    else:
        empty_droplets_distribution = [
            _get_empty_droplets_distribution_from_file(
                (batches_to_file_path[batch], batch),
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



def get_number_of_valid_processes(
    number_of_requested_processes: int, logical_cpu=True
) -> int:
    """Find the number of processes to use in a multiprocess operation.
    This can't be more then the number of processes the compuer have (logical or not) and this can't be negative.

    Args:
        number_of_requested_processes (int): The requested number of processes.

        logical_cpu (bool, optional): Should we use logical number of cpu or physical. Defaults to True.

    Returns:
        int: Max number of valid processes available.
    """
    num_of_available_processes = psutil.cpu_count(logical=logical_cpu)

    # Can't use more then what the user defined, but also can't use something less then 1
    number_of_processes_to_use = max(
        min(number_of_requested_processes, num_of_available_processes), 1
    )

    return number_of_processes_to_use


def calculate_negative_loglikelihood(observed: pd.Series, expected: pd.Series) -> float:
    """Calculate the negative log likelihood of the observed vs expected data.

    Args:
        observed (pd.Series): The observed values in the data.
        expected (pd.Series): The expected values in the data.

    Returns:
        float: Negative log likelihood vlaue of the observerd vs expected datas
    """
    return -expected + np.log(expected) * observed - scipy.special.gammaln(observed)



def _get_empty_droplets_distribution_from_file(
    file_info: tuple[str, str],
    read_file_func: Callable[[str], ad.AnnData],
    max_umi_count_to_be_empty_droplet: int,
) -> pd.DataFrame:
    """Read a droplet file and extract the distribution of all the empty droplets out of it.
    This is being done by taking all the droplets with at most max_umi_count_to_be_empty_droplet umis and aggrigating all of those droplets together.
    Then returning for gene the number of umis seen in all the empty droplets, this later can be used to calculate fraction and frequency of genes distribution in the ambinet noise.

    Args:
        file_info (tuple[str, str]): A tuple with the path to the droplets file and the name of the batch

        read_file_func (Callable[[str], ad.AnnData]): A function that get  a file path of single cell data and read it as AnnData file with the droplet, genes and umi information.
        Examples can be sc.read_h5, sc.read_mtx, and etc.

        max_umi_count_to_be_empty_droplet (int): Droplets with fewer umi count will be considered as empty droplet for distribution calculation.

    Returns:
        pd.DataFrame: Holds the sum of umis per gene in all the empty droplets.
    """
    path_to_droplets_file, batch_name = file_info
    batch_dropelts_adata = read_file_func(os.path.join(path_to_droplets_file))

    # Find all the droplets which are considered as empty base on the threshold
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

    # Aggragate the umis from all the empty droplets.
    umis_per_gene = empty_droplets.sum(axis=0)
    empty_droplets_df = pd.DataFrame(
        data=umis_per_gene.T, index=batch_dropelts_adata.var.index, columns=["umis"]
    )
    empty_droplets_df = empty_droplets_df.groupby(level=0).agg({"umis": "sum"})
    return empty_droplets_df.squeeze()

