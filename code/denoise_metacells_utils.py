"""
Provide a set of functions to handle and correct ambient noise from the metacell process.
- correct_ambient_noise_in_pile_wrapper - A feature_correction function which change the downsampled cells during the metacell derivation and change this derivation to include noise levels consideration.
- denoise_metacells - Posterior reduction of the umi count of each metacell based on its cells components.
"""


from typing import Callable

import anndata as ad
import metacells as mc
import numpy as np
import pandas as pd

import AmbientNoiseFinder


def correct_ambient_noise_in_pile_wrapper(
    ambient_noise_finder: AmbientNoiseFinder.AmbientNoiseFinder,
    cells_adata_with_noise_level_estimations: ad.AnnData,
) -> Callable[[ad.AnnData, ad.AnnData, mc.ut.NumpyMatrix], None]:

    """A wrapper function for the `feature_correction` option in the metacell pipeline.
    Given the AmbientNoiseFinder object which holds the empty droplets distribution per batch and the cells adata with noise levels estimation this will provide a correction
    to the umi count matrix during the first pile calculation -> in turn this should yield a more mix model of metacells.

    Args:
        ambient_noise_finder (AmbientNoiseFinder.AmbientNoiseFinder): Holds the empty droplets information for all the batches information in this dataset.

        cells_adata_with_noise_level_estimations (ad.AnnData): The usuall cells adata file but now includes the noise level estimation.
        This object is received by running ambient_noise_finder.get_cells_adata_with_noise_level_estimations function with the estimation results.

    Returns:
        Callable[[ad.AnnData, ad.AnnData, mc.ut.NumpyMatrix], None]:
        A type of FeatureCorrection function, need to be send to the divide_and_conquer_pipeline function under `feature_correction` variable.
        This will manipulate the downsample cells function before the correlation calculation to generate metacells.
    """
    empty_droplets_umis_per_batch = np.array(
        [
            ambient_noise_finder.batches_empty_droplets_dict[batch]
            for batch in cells_adata_with_noise_level_estimations.obs.batch.values
        ]
    ).squeeze()

    empty_droplets_umis_per_cell_df = pd.DataFrame(
        empty_droplets_umis_per_batch,
        columns=cells_adata_with_noise_level_estimations.var.index,
        index=cells_adata_with_noise_level_estimations.obs.batch.index,
    )

    def correct_ambient_noise_in_pile(
        adata: ad.AnnData, fdata: ad.AnnData, downsampled: mc.ut.NumpyMatrix
    ):
        """
        Correct the ambient noise umi count in the pile by subtracting the umis of each cell by the relevant alpha from the batch and the empty droplets information from the batch.
        We make sure not to drop to a negative umi count by increasing the number of umis we remove from other genes to match the total expected noisy umis.
        This can be though as inflation of the data such that it will hold that:

        E[r_x|X](g) = min(X_g, A * noise_levels * cell_umi_count * empty_droplets_fraction)   s.t. Sum (E[r_x|X]) = noise_levels * cell_umi_count
        With r_x being the noise vector for cell X and A is the infaltion constant to make sure we have indeed explained all the noisy umis.

        Args:
            adata (ad.AnnData): The full annotated data of the cells.

            fdata (ad.AnnData): The feature genes annotated data of the cells (a slice of ``adata``).
            
            downsampled (mc.ut.NumpyMatrix):  A dense matrix of the downsampled UMIs of the features of the cells.
        """
        zero_inflation_factor = 1

        # Convert the empty droplet distribution to fractions.
        current_pile_empty_droplets_umis_df = empty_droplets_umis_per_cell_df.loc[
            fdata.obs.index, fdata.var.index
        ]

        current_pile_empty_droplets_fractions_df = (
            current_pile_empty_droplets_umis_df.div(
                current_pile_empty_droplets_umis_df.sum(axis=1), axis=0
            )
        )

        downsampled_cells_df = pd.DataFrame(
            downsampled, columns=fdata.var.index, index=fdata.obs.index
        )

        number_of_noisy_umis = pd.Series(
            (fdata.obs["batch_estimated_noise"] * np.sum(downsampled, axis=1)).values,
            index=fdata.obs.index,
        )

        # How much noisy umis per gene we curerntly want to remove.
        noisy_umis_per_cells_genes_matrix = np.multiply(
            zero_inflation_factor,
            np.multiply(
                number_of_noisy_umis[:, np.newaxis],
                current_pile_empty_droplets_fractions_df,
            ),
        )

        # Make sure we don't more genes than what we actually have in the matrix.
        valid_noisy_umis_per_cells_genes_matrix = np.minimum(
            noisy_umis_per_cells_genes_matrix, downsampled_cells_df
        )

        number_of_current_valid_noisy_umis = (
            valid_noisy_umis_per_cells_genes_matrix.sum(axis=1)
        )

        number_of_excess_umis = (
            number_of_noisy_umis - number_of_current_valid_noisy_umis
        )

        potential_cells_to_noise_removal = np.where(
            number_of_current_valid_noisy_umis != 0
        )

        while np.any(
            1e-5 < number_of_excess_umis.iloc[potential_cells_to_noise_removal]
        ):
            cells_to_remove_umis_from = number_of_excess_umis.iloc[
                potential_cells_to_noise_removal
            ][1e-5 < number_of_excess_umis.iloc[potential_cells_to_noise_removal]].index

            non_excess_genes = (
                downsampled_cells_df.loc[cells_to_remove_umis_from]
                > noisy_umis_per_cells_genes_matrix.loc[cells_to_remove_umis_from]
            )

            excess_umis = (
                number_of_noisy_umis.loc[cells_to_remove_umis_from]
                - number_of_current_valid_noisy_umis.loc[cells_to_remove_umis_from]
            )

            temp_zero_inflation_factor = np.maximum(
                1,
                1
                + excess_umis
                / (
                    np.sum(
                        current_pile_empty_droplets_fractions_df[non_excess_genes],
                        axis=1,
                    )
                    * np.multiply(number_of_noisy_umis, zero_inflation_factor)
                ),
            )

            temp_zero_inflation_factor[temp_zero_inflation_factor.isna()] = 1

            zero_inflation_factor *= temp_zero_inflation_factor.loc[
                number_of_excess_umis.index
            ]

            noisy_umis_per_cells_genes_matrix = np.multiply(
                zero_inflation_factor[:, np.newaxis],
                np.multiply(
                    number_of_noisy_umis[:, np.newaxis],
                    current_pile_empty_droplets_fractions_df,
                ),
            )

            valid_noisy_umis_per_cells_genes_matrix = np.minimum(
                noisy_umis_per_cells_genes_matrix, downsampled_cells_df
            )

            number_of_current_valid_noisy_umis = (
                valid_noisy_umis_per_cells_genes_matrix.sum(axis=1)
            )

            number_of_excess_umis = (
                number_of_noisy_umis - number_of_current_valid_noisy_umis
            )

            potential_cells_to_noise_removal = np.where(
                number_of_current_valid_noisy_umis != 0
            )  # zero noisy umis

        downsampled -= valid_noisy_umis_per_cells_genes_matrix

    return correct_ambient_noise_in_pile


def denoise_metacells(
    cells_adata_with_noise_level_estimations: ad.AnnData,
    metacells_ad: ad.AnnData,
    ambient_noise_finder: AmbientNoiseFinder.AmbientNoiseFinder,
    valid_obs=["grouped", "pile", "candidate"],
    valid_var=[
        "forbidden_gene",
        "pre_feature_gene",
        "feature_gene",
        "top_feature_gene",
    ],
    blacklist_obs=["umap_x", "umap_y"],
    blacklist_var=[],
):
    """Go over each metacell and subtract the expected noisy umis based on all the relevant cells. Aggrigating and removing cells noise based on batch and umi depth bin.
    Here we make sure not to have negative umis but we won't add those umis to someplace else because we expect the aggregation of cells in each metacell to have strong enough
    statistical power to use method and still have good approximation.

    Args:
        cells_adata_with_noise_level_estimations (ad.AnnData): The usuall cells adata file but now includes the noise level estimation.
        This object is received by running ambient_noise_finder.get_cells_adata_with_noise_level_estimations function with the estimation results.

        metacells_ad (ad.AnnData): The output of divide_and_conquer_pipeline function, an addata object which represent the metacells generated from the cells_adata_with_noise_level_estimations object.

        ambient_noise_finder (AmbientNoiseFinder.AmbientNoiseFinder): Holds the empty droplets information for all the batches information in this dataset.

        valid_obs (list, optional): List of valid columns in the obs object which we want to pass to the new addata. Defaults to ["grouped", "pile", "candidate"].

        valid_var (list, optional): List of valid columns in the var object which we want to pass to the new addata. Defaults to [ "forbidden_gene", "pre_feature_gene", "feature_gene", "top_feature_gene", ].

        blacklist_obs (list, optional): List of columns in the obs object to remove. Defaults to ["umap_x", "umap_y"].

        blacklist_var (list, optional): List of columns in the var object to remove. Defaults to [].

    Returns:
        _type_: _description_
    """
    denoise_metacells_df = mc.ut.get_vo_frame(metacells_ad).copy()

    cells_info_by_metacells_batch_umi_depth = (
        cells_adata_with_noise_level_estimations.obs.groupby(
            ["batch", "umi_depth_bin", "metacell"]
        ).agg({"effective_umi_depth": "sum", "batch_estimated_noise": "mean"})
    )

    for batch_name in ambient_noise_finder.batches_empty_droplets_dict.keys():
        batch_empty_droplets_fraction = (
            ambient_noise_finder.batches_empty_droplets_dict[batch_name].loc[
                metacells_ad.var.index
            ]
            / ambient_noise_finder.batches_empty_droplets_dict[batch_name].sum()
        )

        for umi_depth_bin in cells_adata_with_noise_level_estimations.obs[
            cells_adata_with_noise_level_estimations.obs.batch == batch_name
        ].umi_depth_bin.unique():

            metacells_umis_for_batch_and_umi_depth_bin = (
                cells_info_by_metacells_batch_umi_depth.loc[
                    batch_name, umi_depth_bin
                ].effective_umi_depth
                * cells_info_by_metacells_batch_umi_depth.loc[
                    batch_name, umi_depth_bin
                ].batch_estimated_noise
            )

            noise_per_mc = (
                metacells_umis_for_batch_and_umi_depth_bin.effective_umi_depth.values.reshape(
                    -1, 1
                )
                * batch_empty_droplets_fraction.T.values
            )
            denoise_metacells_df.iloc[
                metacells_umis_for_batch_and_umi_depth_bin.index
            ] -= noise_per_mc

    # Make sure we don't have zeros in our matrix: max(0, obs-noise)
    denoise_metacells_df[denoise_metacells_df < 0] = 0

    exist_vs_valid_obs = list(
        set(metacells_ad.obs.columns) - set(valid_obs) - set(blacklist_obs)
    )

    exist_vs_valid_var = list(
        set(metacells_ad.var.columns) - set(valid_var) - set(blacklist_var)
    )

    if len(exist_vs_valid_obs):
        print(
            "Found non valid obs and didn't passed them, insert manualy to pass:\n%s"
            % (" ".join(exist_vs_valid_obs))
        )

    if len(exist_vs_valid_var):
        print(
            "Found non valid var and didn't passed them, insert manualy to pass:\n%s"
            % (" ".join(exist_vs_valid_var))
        )

    valid_obs = metacells_ad.obs.columns & valid_obs
    valid_var = metacells_ad.var.columns & valid_var
    denoise_metacells_ad = ad.AnnData(
        X=denoise_metacells_df.to_numpy(),
        obs=metacells_ad.obs[valid_obs],
        var=metacells_ad.var[valid_var],
    )
    return denoise_metacells_ad
