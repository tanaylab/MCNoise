"""
Provide a set of functions to handle and correct ambient noise from the metacell process.
- generate_denoised_metacells:
     Posterior reduction of the umi count of each metacell based on its cells components.
"""


from typing import Callable, Union, Tuple

import utilities
import anndata as ad
import metacells as mc
import numpy as np
import pandas as pd
import scipy

import multiprocessing

import ambient_logger
import warnings
import utilities

# Two globals to be used when calclating the denoise version of each batch
global cells_adata_with_noise_level_estimations_g
global batches_empty_droplets_dict_g



def generate_denoised_metacells(cells_anndata_with_noise_level_estimations: ad.AnnData,
                                         batches_empty_droplets_dict: dict[str, pd.Series],
                                         denoised_number_of_processed: int = 20,
                                         ) -> Tuple[ad.AnnData, ad.AnnData]:

    """
    Denoise the cells anndata matrix, run divide and conquer and collect metacells on the clean anndata and return the metacells anndata and the cells anndata with the metacells locations.

    :param cells_anndata_with_noise_level_estimations:
        The full cells adata file including the noise level estimation.
        This object is received by running ambient_noise_finder.get_cells_adata_with_noise_level_estimations function with the estimation results.
    :type cells_anndata_with_noise_level_estimations: ad.AnnData

    :param batches_empty_droplets_dict: Mapping between the batch name and the empty droplets distribution across the genes.
    :type batches_empty_droplets_dict: dict[str, pd.Series]

    :param denoised_number_of_processed: Number of processes to work together when calculating the denoise batch matrices, default to 20
    :type denoised_number_of_processed: int

    :return: The original cells anndata with the mapping to the clean metacells and the metacell anndata file after denoising of the metacells umi count.
    :rtype: Tuple[ad.AnnData, ad.AnnData]
    """
    logger = ambient_logger.logger()
    cells_anndata_with_noise_level_estimations = cells_anndata_with_noise_level_estimations.copy()
    
    number_of_process = utilities.get_number_of_processes_to_use(
        number_of_requested_processes=denoised_number_of_processed,
        number_of_tasks=len(
            cells_anndata_with_noise_level_estimations.obs.batch.unique()
        ),
    )

    logger.info("Starting to deniose entire cell matrix")
    denoised_cells_matrix = _get_denoise_cells_matrix_csr(
        cells_anndata_with_noise_level_estimations,
        batches_empty_droplets_dict,
        number_of_process,
    )

    rounded_denoised_cells_matrix = utilities.stochastic_round_sparse(denoised_cells_matrix).astype(np.float32)

    denoised_cells_ad = ad.AnnData(
        rounded_denoised_cells_matrix,
        obs=cells_anndata_with_noise_level_estimations.obs,
        var=cells_anndata_with_noise_level_estimations.var,
    )
    
    logger.info("Finish deniosing entire cell matrix")

   
    
    return denoised_cells_ad


def _get_denoise_cells_matrix_csr(
    cells_anndata_with_noise_level_estimations: ad.AnnData,
    batches_empty_droplets_dict: dict[str, pd.Series],
    number_of_process: int,
) -> scipy.sparse._csr.csr_matrix:
    """
    Go over all the cells matrices - one per batch, and denoise it using `_denoise_single_batch_matrix`.
    After having the denoise matrices, concat it again to a full matrix with all the cells.

    :param cells_anndata_with_noise_level_estimations:
        The full cells adata file including the noise level estimation.
        This object is received by running ambient_noise_finder.get_cells_adata_with_noise_level_estimations function with the estimation results.
    :type cells_anndata_with_noise_level_estimations: ad.AnnData

    :param batches_empty_droplets_dict: Mapping between the batch name and the empty droplets distribution across the genes.
    :type batches_empty_droplets_dict: dict[str, pd.Series]

    :param number_of_process: Number of processes to work together when calculating the denoise batch matrices, default to 20
    :type number_of_process: int

    :return: The denoise cells matrix, reorder based on the original order
    :rtype: np.ndarray
    """
    global cells_adata_with_noise_level_estimations_g
    global batches_empty_droplets_dict_g

    cells_adata_with_noise_level_estimations_g = (  # type: ignore
        cells_anndata_with_noise_level_estimations
    )
    batches_empty_droplets_dict_g = batches_empty_droplets_dict  # type: ignore

    batches_list = list(cells_anndata_with_noise_level_estimations.obs.batch.unique())
    
    with multiprocessing.Pool(processes=number_of_process) as multiprocess_pool:
        denoised_matrices = multiprocess_pool.map(
            _denoise_single_batch_matrix_csr, batches_list
        )

    denoise_cells_ad_list = []
    for batch_index in range(len(batches_list)):
        denoise_matrix, denoise_matrix_cells_order = denoised_matrices[batch_index]
        denoise_cells_ad_list.append(
            ad.AnnData(denoise_matrix,obs=pd.DataFrame(denoise_matrix_cells_order, index=denoise_matrix_cells_order, columns=["cell_id"]),var=cells_anndata_with_noise_level_estimations.var)
        )

    combined_ad = ad.concat(denoise_cells_ad_list)[
        cells_anndata_with_noise_level_estimations.obs.index, :
    ]

    return combined_ad.X


def _denoise_single_batch_matrix_csr(batch: str) -> Tuple[scipy.sparse._csr.csr_matrix, pd.Index]:
    """
    Calculate the expected niose of each cells, based on it's batch information.
    Using the calculated alpha, the noise estimation distribution from the empty droplets and the cell size - we can calculate the expected noise per cell.
    After removing this we will have negative values in some cells-genes, to perform non-zero inflation we will recollect those UMIs and redistribute them between
    the different cells and genes, based on the amount of expected residual of noisy UMIs.
    We perform this process until no more noisy UMIs are there to be distributed

    :param batch: The name of the batch currently cleaning
    :type batch: str

    :return: A clean version of the cells matrix and the order of the cells
    :rtype: Tuple[np.ndarray, pd.Index]
    """
    global cells_adata_with_noise_level_estimations_g
    global batches_empty_droplets_dict_g

    logger = ambient_logger.logger()
    logger.info("Starts to denoise batch: %s" % batch)

    orig_err = np.geterr()
    np.seterr(divide="ignore", invalid="ignore")
    warnings.simplefilter('ignore',scipy.sparse.SparseEfficiencyWarning)
    batch_cells_anndata = cells_adata_with_noise_level_estimations_g[cells_adata_with_noise_level_estimations_g.obs.batch == batch] # type: ignore
    batch_cells_matrix = batch_cells_anndata.X.copy()
    batch_empty_drouplets_distribution = batches_empty_droplets_dict_g[batch]/ batches_empty_droplets_dict_g[batch].sum() # type: ignore
    
    # remove the expected value from each cell
    expected_noisy_umis_per_cell_gene = scipy.sparse.csr_matrix((batch_cells_anndata.obs.batch_estimated_noise * batch_cells_anndata.obs.umi_depth).values[:,np.newaxis] * batch_empty_drouplets_distribution.values)

    expected_noisy_umis_from_valid_genes = (batch_cells_matrix != 0).multiply(expected_noisy_umis_per_cell_gene)
    expected_noisy_umis_from_invalid_genes = expected_noisy_umis_per_cell_gene - expected_noisy_umis_from_valid_genes
    batch_cells_matrix -= expected_noisy_umis_from_valid_genes 
    batch_cells_matrix.eliminate_zeros()

    # Get exceeded infomation
    residual_noisy_umis_per_cells_genes = batch_cells_matrix.minimum(0).multiply(-1)
    residual_noisy_umis_per_genes = (expected_noisy_umis_from_invalid_genes + residual_noisy_umis_per_cells_genes).sum(axis=0)
    residual_noisy_umis_per_cells = (expected_noisy_umis_from_invalid_genes + residual_noisy_umis_per_cells_genes).sum(axis=1)

    # clipped negative umis after the collection of residual
    batch_cells_matrix[batch_cells_matrix < 0] = 0
    batch_cells_matrix.eliminate_zeros()

    while np.any(residual_noisy_umis_per_genes > 1):
        # Get indices of genes with too many noisy umis to remove --> this mean we will have to move those umis to some other genes baesed on the ambient noise
        unvalid_genes_to_remove_from =np.array(residual_noisy_umis_per_genes > batch_cells_matrix.sum(axis=0)).reshape(-1)

        # Get the number of umis we need to move from one gene to another
        residual_genes_to_move = (residual_noisy_umis_per_genes[:,unvalid_genes_to_remove_from] - batch_cells_matrix.sum(axis=0)[:,unvalid_genes_to_remove_from] ).sum()

        # clipped the number of UMIs we can remove in those genes to the existing ones in the current matrix
        np.putmask(residual_noisy_umis_per_genes, unvalid_genes_to_remove_from, batch_cells_matrix.sum(axis=0))

        # Move the residual genes with no other match to other genes, based on the ambient noise distribution
        ambient_noise_distribution_given_valid_genes = batch_empty_drouplets_distribution.values[~unvalid_genes_to_remove_from] / batch_empty_drouplets_distribution.values[~unvalid_genes_to_remove_from].sum()
        residual_noisy_umis_per_genes[:,~unvalid_genes_to_remove_from] += ambient_noise_distribution_given_valid_genes * residual_genes_to_move

        # Get distribution of umis per gene per cell we need to remove, after normalizing to the genes we can actually remove from
        redistribute_matrix = (batch_cells_matrix != 0).multiply(residual_noisy_umis_per_genes)
        redistribute_matrix_normmed = redistribute_matrix.multiply(1/redistribute_matrix.sum(axis=1)).tocsr()
        redistribute_matrix_normmed[np.array(redistribute_matrix.sum(axis=1) == 0).reshape(-1),:] = 0   # handle the multiplication by inf from previous line

        # Multiple by the number of noisy UMIs per cell
        redistribute_matrix_normmed_with_genes = redistribute_matrix_normmed.multiply(residual_noisy_umis_per_cells).tocsr()

        # Make sure we don't remove more than we have in the cells matrix
        expected_noise_matrix = redistribute_matrix_normmed_with_genes.minimum(batch_cells_matrix)

        # residuals
        residual_noisy_umis_per_cells_genes = redistribute_matrix_normmed_with_genes - expected_noise_matrix
        batch_cells_matrix -= expected_noise_matrix

        residual_noisy_umis_per_genes = residual_noisy_umis_per_cells_genes.sum(axis=0)
        residual_noisy_umis_per_cells = residual_noisy_umis_per_cells_genes.sum(axis=1)
        batch_cells_matrix[batch_cells_matrix < 0] = 0
        batch_cells_matrix.eliminate_zeros()


    np.seterr(divide=orig_err["divide"], invalid=orig_err["invalid"])
    warnings.simplefilter('default',scipy.sparse.SparseEfficiencyWarning)
    logger.info("Finish denoising batch: %s" % batch)

    return batch_cells_matrix, batch_cells_anndata.obs.index
