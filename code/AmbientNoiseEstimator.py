"""
AmbientNoiseEstimator

Object that wrap all the needed functions to estimate the ambient noise levels in a given dataset.
Based on the notion that the observed umis are mixture of the native expressions of the cell state + the noise distribution of a given batch.
Build around the data object `AmbientNoiseFinder` which holds the information about the cells, metacells and batch empty droplets. 
We can estimate both the noise levels and native expressions based on the noise-prone metacells-genes clusters which were discovered using AmbientNoiseFinder.
"""

from typing import Union
import metacells as mc
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from AmbientNoiseFinder import AmbientNoiseFinder
from EstimationResults import NoiseNativeExpressionEstimation
from RGlmSolver import RGlmSolver
from utilities import *


class AmbientNoiseEstimator(object):
    def __init__(
        self,
        ambient_noise_finder: AmbientNoiseFinder,
    ) -> None:
        """
        Store the ambient noise finder object, which holds all the information about the dataset - will be later used in `estimate_noise_levels` for estimation.
        Initialized the GLM solver and a basic template to store the information about noisy oriented metacells-genes clusters.

        :param ambient_noise_finder: Hold all the information needed to estimate a dataset's noise, including the cells, metacells, and empty droplets information.
        :type ambient_noise_finder: AmbientNoiseFinder
        """
        self.ambient_noise_finder = ambient_noise_finder

        self.cells_genes_clusters_info_template = (
            self._create_cells_genes_clusters_info_template()
        )

        # Initialized the a R based GLM solver to handle and solve the equations.
        self.r_glm_solver = RGlmSolver()

    def estimate_noise_levels(
        self,
        max_relative_expression_for_metacells_genes_to_use: float,
        number_of_steps: int = 10,
        use_small_metacells_clusters: bool = False,
        use_small_genes_clusters: bool = False,
        max_valid_observed_umis_per_cell: int = None,
        min_expected_umi_threshold: int = 100,
        min_number_of_pgm_clusters_per_batch: int = 3,
        min_number_of_batches_per_pgm_cluster: int = 3,
        number_of_cv: int = 10,
        umi_depth_bins_to_calculate: Union[None, list[int]] = None,
    ) -> NoiseNativeExpressionEstimation:
        """
        Estimate the noise levels and the native expression fraction of genes in a metacell cluster.
        This is being calculated using the equations: observed = #umis * (native_expression_fraction + noise_estimation * empty_droplet_fraction)
        Using a Poission loss to fit the counting information.

        Computation:
            1. Init an NoiseNativeExpressionEstimation object with the given user variables to store the current run information.

            2. Use `_extract_noisy_oriented_clusters_data` to collect the umi information from all the metacells-genes which might be considered as noisy
                This mean that they have relative expression below `max_relative_expression_for_metacells_genes_to_use`, split them to differnet steps based on `number_of_steps`.
                Use or remove small metacells\genes clusters based on `use_small_metacells_clusters` and `use_small_genes_clusters` values.
                Also going to remove cells with umis above `max_valid_observed_umis_per_cell` or 4 times the median of the cells in case None was provided.

            3. For each umi depth bin and relative expression step extract the matching information, convert it to equation format ready for GLM solver.
               Filter out batches and metacells-clusters using `_get_valid_batches_and_cells_genes_clusters`, this will take into account the `min_number_of_pgm_clusters_per_batch`
               and `min_number_of_batches_per_pgm_cluster` values the user provided.

            4. Combine the different umi depths equation in every step and perform the estimation for `number_of_cv`, each fold will be of size 90% of the equations.

            5. Add the current step estimation into the NoiseNativeExpressionEstimation object and return it once finished.


        :param max_relative_expression_for_metacells_genes_to_use:
            An upper threshold for metacells-genes clusters, which are considered noisy.
            The `metacells_genes_clusters_median_relative_expression_to_max_df` from the ambient_noise_finder state the relative expression between each metacell-gene cluster to
            the most expressed metacell cluster for the same gene cluster.
            Using it, we order all the clusters from the most noise-oriented one to the last, using all below the given value.
        :type max_relative_expression_for_metacells_genes_to_use: float

        :param number_of_steps: Number of estimation steps, in each step we add more metacell-genes clusters so higher number of steps mean adding less cluster in each step.
                                defaults to 10.
        :type number_of_steps: int, optional

        :param use_small_metacells_clusters: Is the algorithm allowed to use the small metacell clusters from the ambient_noise_finder, defaults to False.
        :type use_small_metacells_clusters: bool, optional

        :param use_small_genes_clusters: Is the algorithm allowed to use the small genes clusters from the ambient_noise_finder,  defaults to False.
        :type use_small_genes_clusters: bool, optional

        :param max_valid_observed_umis_per_cell: Upper threshold for the number of umis per specific cell, if too much we assume this isn't noise, defaults to None.
                                                 Using None will yield a 4 * median of cells.
        :type max_valid_observed_umis: int, optional

        :param min_expected_umi_threshold: Metacells-genes clusters with an expected number of umis lower then this for 100% noise will be removed from calculation.
                                           This is being used to make sure we have enough information in each equation to make precise calculation, defaults to 100
        :type min_expected_umi_threshold: int, optional

        :param min_number_of_pgm_clusters_per_batch: Batches with less than this amount of different metacell-genes clusters(pgm) won't be calculated due to few data points.
                                                     defaults to 3
        :type min_number_of_pgm_clusters_per_batch: int, optional

        :param min_number_of_batches_per_pgm_cluster: Metacell-genes clusters (pgm) with less than this amount of batches where they appear will be removed.
                                                      defaults to 3
        :type min_number_of_batches_per_pgm_cluster: int, optional

        :param number_of_cv: Number of cross validation folds to use on the given equation set, defaults to 10.
        :type number_of_cv: int, optional

        :param umi_depth_bins_to_calculate: If given, will perform the estimation only for those umi bins, defaults to None.
        :type umi_depth_bins_to_calculate: Union[None, list[int]], optional

        :return: An estimation of the noise and native expression across the different steps for all the umi depth bins.
        :rtype: NoiseNativeExpressionEstimation
        """
        estimation_obj = NoiseNativeExpressionEstimation(
            max_relative_expression_for_metacells_genes_to_use=max_relative_expression_for_metacells_genes_to_use,
            number_of_steps=number_of_steps,
            min_expected_umi_threshold=min_expected_umi_threshold,
            min_number_of_pgm_clusters_per_batch=min_number_of_pgm_clusters_per_batch,
            min_number_of_batches_per_pgm_cluster=min_number_of_batches_per_pgm_cluster,
        )

        umi_depth_bins = set(range(self.ambient_noise_finder.umi_depth_number_of_bins))
        if umi_depth_bins_to_calculate:
            umi_depth_bins = umi_depth_bins & set(umi_depth_bins_to_calculate)

        metacells_genes_cluster_relative_expression_df = (
            self.ambient_noise_finder.metacells_genes_clusters_median_relative_expression_to_max_df
        )

        valid_relative_expression_values = (
            metacells_genes_cluster_relative_expression_df.values[
                metacells_genes_cluster_relative_expression_df.values
                < max_relative_expression_for_metacells_genes_to_use
            ]
        )

        noisy_cells_genes_clusters = self._extract_noisy_oriented_clusters_data(
            max_relative_expression_for_metacells_genes_to_use,
            number_of_steps,
            use_small_metacells_clusters=use_small_metacells_clusters,
            use_small_genes_clusters=use_small_genes_clusters,
            max_valid_observed_umis=max_valid_observed_umis_per_cell,  # type: ignore
            min_expected_umi_threshold=min_expected_umi_threshold,
        )

        for step, current_relative_expression in enumerate(
            tqdm(
                np.quantile(
                    valid_relative_expression_values, np.linspace(0, 1, number_of_steps)
                )
            )
        ):

            cells_genes_clusters_for_current_max_relative_expression = (
                noisy_cells_genes_clusters[noisy_cells_genes_clusters.step <= step]
            )

            umi_depth_equations_list = []
            for umi_depth_bin in umi_depth_bins:
                current_bin_cells_genes_clusters = (
                    cells_genes_clusters_for_current_max_relative_expression.loc[
                        umi_depth_bin
                    ]
                )

                cells_genes_clusters_equations = (
                    self._format_cells_genes_clusters_dataframe_as_equations(
                        current_bin_cells_genes_clusters
                    )
                )

                (
                    valid_batches,
                    valid_cells_genes_clusters,
                ) = self._get_valid_batches_and_cells_genes_clusters(
                    cells_genes_clusters=current_bin_cells_genes_clusters,
                    min_number_of_pgm_clusters_per_batch=min_number_of_pgm_clusters_per_batch,
                    min_number_of_batches_per_pgm_cluster=min_number_of_batches_per_pgm_cluster,
                )

                if len(valid_batches) == 0 or len(valid_cells_genes_clusters) == 0:
                    # TODO: add logging
                    continue

                valid_equations = self._arrange_and_filter_equations_based_on_valid_batches_and_clusters(
                    noisy_oriented_clusters_equations=cells_genes_clusters_equations,
                    valid_batches=valid_batches,
                    valid_genes_cells_clusters=valid_cells_genes_clusters,
                )

                # Add label to the different batches based on the umi depth bins they represent.
                # This will allow different estimation for different bins but the same for the genes-cells pairs.
                columns_labels_with_bin_reference = [
                    "%s_umi_depth_bin_%s" % (column, umi_depth_bin)
                    if column in valid_batches
                    else column
                    for column in valid_equations.columns
                ]
                valid_equations.columns = columns_labels_with_bin_reference
                umi_depth_equations_list.append(valid_equations)

            if len(umi_depth_equations_list) == 0:
                # TODO: add logging
                continue

            # Combine all the equations from the different umi depth bins
            all_umi_depth_equations = pd.concat(umi_depth_equations_list)
            all_umi_depth_equations = all_umi_depth_equations.fillna(0)

            noise_native_expression_estimation = self._solve_equations(
                equations=all_umi_depth_equations,
                number_of_cross_validation_folds=number_of_cv,
            )

            estimation_obj.add_estimation_step(
                noise_native_expression_estimation=noise_native_expression_estimation,
                equations=umi_depth_equations_list,
                step=step,
                step_relative_expression=current_relative_expression,
            )

        return estimation_obj

    def _create_cells_genes_clusters_info_template(self) -> pd.DataFrame:
        """
        Generate and calculate all the information needed for the cells-genes cluster.
        This will split the cells information based on the batch, metacell of origion, gene cluster and umi depth bin to provide easy access and combination of them in the future.

        :return: A representation of all the cells-genes clusters information, which includes:
                 The observed umis for this cells-genes, the total umis count for those cells, the batch of the cells, the empty droplets fractions for this gene cluster,
                 the cells clusters id and the umi depth bin for those cells.
        :rtype: pd.DataFrame
        """
        cells_genes_cluster_template = pd.DataFrame(
            index=self.ambient_noise_finder.cells_adata.obs.index,
            columns=[
                "observed",
                "total_umis",
                "batch",
                "metacell",
                "empty_droplets_fraction",
                "cells_cluster",
                "umi_depth_bin",
            ],
        )

        cells_genes_cluster_template[
            "metacell"
        ] = self.ambient_noise_finder.cells_adata.obs.metacell

        cells_genes_cluster_template[
            "total_umis"
        ] = self.ambient_noise_finder.cells_adata.obs.umi_depth

        cells_genes_cluster_template[
            "batch"
        ] = self.ambient_noise_finder.cells_adata.obs.batch

        cells_genes_cluster_template[
            "cells_cluster"
        ] = self.ambient_noise_finder.cells_adata.obs.cells_cluster

        cells_genes_cluster_template[
            "umi_depth_bin"
        ] = self.ambient_noise_finder.cells_adata.obs.umi_depth_bin

        return cells_genes_cluster_template

    def _extract_noisy_oriented_clusters_data(
        self,
        max_relative_expression_for_metacells_genes_to_use: float,
        number_of_steps: int,
        use_small_metacells_clusters: bool,
        use_small_genes_clusters: bool,
        max_valid_observed_umis: int,
        min_expected_umi_threshold: int,
    ) -> pd.DataFrame:
        """
        Go over all the valid metacells-genes clusters pairs which are deemed to be noisy oriented (below the max_relative_expression_for_metacells_genes_to_use)
        and collect all the metacells-genes cluster information needed for later estimations. For each pair, split the data based on batch and umi depth bin.
        All cells will match the following criteria:
                - They have relative expression below `max_relative_expression_for_metacells_genes_to_use`, split to differnet steps based on `number_of_steps`.
                - Use or remove small metacells\genes clusters based on `use_small_metacells_clusters` and `use_small_genes_clusters` values.
                - Remove cells with umis above `max_valid_observed_umis_per_cell` or 4 times the median of the cells in case None was provided.

        :param max_relative_expression_for_metacells_genes_to_use:
            An upper threshold for metacells-genes clusters, which are considered noisy.
            The `metacells_genes_clusters_median_relative_expression_to_max_df` from the ambient_noise_finder state the relative expression between each metacell-gene cluster to
            the most expressed metacell cluster for the same gene cluster.
            Using it, we order all the clusters from the most noise-oriented one to the last, using all below the given value.
        :type max_relative_expression_for_metacells_genes_to_use: float

        :param number_of_steps: Number of estimation steps, in each step we add more metacell-genes clusters so higher number of steps mean adding less cluster in each step.
        :type number_of_steps: int

        :param use_small_metacells_clusters: Is the algorithm allowed to use the small metacell clusters from the ambient_noise_finder.
        :type use_small_metacells_clusters: bool

        :param use_small_genes_clusters: Is the algorithm allowed to use the small genes clusters from the ambient_noise_finder.
        :type use_small_genes_clusters: bool

        :param max_valid_observed_umis: Upper threshold for the number of umis per specific cell, if too much we assume this isn't noise, defaults to None.
                                        Using None will yield a 4 * median of cells.
        :type max_valid_observed_umis: int

        :param min_expected_umi_threshold: Metacells-genes clusters with an expected number of umis lower then this for 100% noise will be removed from calculation.
                                           This is being used to make sure we have enough information in each equation to make precise calculation.
        :type min_expected_umi_threshold: int

        :return: An aggragated version of cells_genes_clusters_info_template after filtering based on the user paramaters, this is ready to be converted to equations.
        :rtype: pd.DataFrame
        """

        cells_genes_clusters = []
        already_calculated_clusters = []

        metacells_genes_clusters_relative_expression_df = (
            self.ambient_noise_finder.metacells_genes_clusters_median_relative_expression_to_max_df
        )

        valid_relative_expression_values = (
            metacells_genes_clusters_relative_expression_df.values[
                metacells_genes_clusters_relative_expression_df.values
                < max_relative_expression_for_metacells_genes_to_use
            ]
        )

        for step, current_relative_expression in enumerate(
            np.quantile(
                valid_relative_expression_values, np.linspace(0, 1, number_of_steps)
            )
        ):

            combined_metacells_genes_cluster_noisy_information = pd.DataFrame()

            # Get the set of valid metacells clusters and genes clusters
            metacell_cluster_list, gene_cluster_list = np.where(
                metacells_genes_clusters_relative_expression_df
                <= current_relative_expression
            )

            for j, metacells_cluster in enumerate(metacell_cluster_list):
                genes_cluster = metacells_genes_clusters_relative_expression_df.columns[
                    gene_cluster_list[j]
                ]

                # If this metacell clusters is consider small cluster don't use it unless the user stated that we should use it.
                if (
                    metacells_cluster
                    in self.ambient_noise_finder.small_metacells_clusters
                    and not use_small_metacells_clusters
                ):
                    continue

                if (
                    genes_cluster in self.ambient_noise_finder.small_genes_clusters
                    and not use_small_genes_clusters
                ):
                    continue

                label = "Pgm_%s_in_%s" % (genes_cluster, metacells_cluster)
                if label in already_calculated_clusters:
                    continue

                already_calculated_clusters.append(label)

                # Extract the specific pair information
                metacells_genes_cluster_noisy_information = (
                    self._get_metacells_genes_cluster_noisy_information(
                        metacells_cluster, genes_cluster
                    )
                )

                # Def the upper limit of observed umis per cells, either by the user definition or from the data itself.
                max_valid_observed_umis = (
                    (metacells_genes_cluster_noisy_information.observed.median() + 1)
                    * 4
                    if max_valid_observed_umis is None
                    else max_valid_observed_umis
                )

                metacells_genes_cluster_noisy_information = (
                    metacells_genes_cluster_noisy_information[
                        metacells_genes_cluster_noisy_information.observed
                        <= max_valid_observed_umis
                    ]
                )

                # Combine the information from all the metacell-genes cluster pair based on origin batch and umi depth bin
                combined_metacells_genes_cluster_noisy_information = (
                    metacells_genes_cluster_noisy_information.groupby(
                        ["umi_depth_bin", "batch"]
                    ).agg({"observed": "sum", "expected": "sum", "total_umis": "sum"})
                )

                # Filter pairs without observed umis or with small number of expected umis
                combined_metacells_genes_cluster_noisy_information = (
                    combined_metacells_genes_cluster_noisy_information[
                        (
                            combined_metacells_genes_cluster_noisy_information.expected
                            >= min_expected_umi_threshold
                        )
                        & (
                            combined_metacells_genes_cluster_noisy_information.observed
                            > 0
                        )
                    ]
                )

                # Add relevant labels to identify this pair
                combined_metacells_genes_cluster_noisy_information[
                    "relative_expression"
                ] = current_relative_expression

                combined_metacells_genes_cluster_noisy_information["step"] = step

                combined_metacells_genes_cluster_noisy_information[
                    "genes_metacells_cluster"
                ] = label

                combined_metacells_genes_cluster_noisy_information = (
                    combined_metacells_genes_cluster_noisy_information.set_index(
                        "genes_metacells_cluster", append=True
                    )
                )

                combined_metacells_genes_cluster_noisy_information.reorder_levels(
                    ["umi_depth_bin", "genes_metacells_cluster", "batch"]
                )

            cells_genes_clusters.append(
                combined_metacells_genes_cluster_noisy_information
            )

        return pd.concat(cells_genes_clusters)

    def _get_metacells_genes_cluster_noisy_information(
        self, metacells_cluster: int, genes_cluster: int
    ) -> pd.DataFrame:
        """
        Fill in specific metacells-genes cluster pair information in a cells_genes_clusters_pair template.

        :param metacells_cluster: The id of the metacells cluster.
        :type metacells_cluster: int

        :param genes_cluster: The id of the genes cluster.
        :type genes_cluster: int

        :return: Hold all the relevant information for a specific genes-metacells cluster.
                 This will later be used to generate the equations for the solver based on the observed umis, the total umis, and the empty droplets fractions information.
        :rtype: pd.DataFrame
        """
        cluster_metacells = self.ambient_noise_finder.metacells_adata.obs[
            self.ambient_noise_finder.metacells_adata.obs.metacells_cluster
            == metacells_cluster
        ].index

        cluster_genes = self.ambient_noise_finder.metacells_adata.var[
            self.ambient_noise_finder.metacells_adata.var.genes_cluster == genes_cluster
        ].index

        cells_genes_clusters = self.cells_genes_clusters_info_template.copy()
        cells_genes_clusters = cells_genes_clusters[
            np.in1d(cells_genes_clusters.metacell, cluster_metacells.astype(int))
        ]

        # Add the specific empty droplets fraction information based on the batch and the genes cluster we work with.
        for (
            batch
        ) in self.ambient_noise_finder.empty_droplet_genes_cluster_fraction.index:
            cells_genes_clusters.loc[
                cells_genes_clusters.batch == batch, "empty_droplets_fraction"
            ] = self.ambient_noise_finder.empty_droplet_genes_cluster_fraction.loc[
                batch, genes_cluster
            ]

        cells_genes_clusters["observed"] = np.squeeze(
            np.asarray(
                self.ambient_noise_finder.cells_adata[
                    cells_genes_clusters.index, cluster_genes
                ].X.sum(axis=1)
            )
        )
        cells_genes_clusters["expected"] = (
            cells_genes_clusters["empty_droplets_fraction"]
            * cells_genes_clusters["total_umis"]
        )

        return cells_genes_clusters

    def _format_cells_genes_clusters_dataframe_as_equations(
        self, cells_genes_clusters: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert a data frame that holds all the data about a specific genes-cells cluster pair, batch, and umi depth to an equation format data frame ready for estimation.
        This is mostly rearranging the data differently to represent the equation:

            observeed = (batch_noise_levels * empty_droplets_fraction + native_expression) * total_umis

        :param cells_genes_clusters: One line for each combination of genes-metacells cluster pair, batch, umi depth bins.
        :type cells_genes_clusters: pd.DataFrame

        :return: An equation-based data frame to represent the noisy data in a solvable manner for the next steps.
                Each row is a different equation based on the cells-genes cluster, batch, and umi depth.
        :rtype: pd.DataFrame
        """
        number_of_equations = cells_genes_clusters.shape[0]

        batches_labels = cells_genes_clusters.index.get_level_values(0).unique()
        cells_genes_clusters_labels = cells_genes_clusters.index.get_level_values(
            1
        ).unique()

        observed_umis_df = pd.DataFrame(
            cells_genes_clusters["observed"].values, columns=["observed"]
        )

        # Generate a dataframe with a repeated vector of native expression, in each row we have the native expression coefficient (total umis) of this specific cell-genes cluster
        native_expression_df = pd.DataFrame(
            np.repeat(
                cells_genes_clusters.total_umis.values,
                len(cells_genes_clusters_labels),
            ).reshape(number_of_equations, len(cells_genes_clusters_labels)),
            columns=cells_genes_clusters_labels,
        )
        # Go over all the rows(cells_genes_clusters_pair), remove values from true native columns which doesn't match the row pgm
        for cells_genes_clusters_id in cells_genes_clusters_labels:
            native_expression_df.loc[
                cells_genes_clusters.index.get_level_values(1) == cells_genes_clusters_id,
                native_expression_df.columns != cells_genes_clusters_id,
            ] = 0

        # Generate a dataframe with a repeated vector of noise levels estimation, in each row we have the noise levels coefficient (total umis * empty droplets fraction) of this specific cell-genes pairs
        noise_level_estimation_df = pd.DataFrame(
            np.repeat(
                cells_genes_clusters.expected.values, len(batches_labels)
            ).reshape(number_of_equations, len(batches_labels)),
            columns=batches_labels,
        )

        # Go over all the rows(batch), remove values from batch columns which doesn't match the row batch
        for batch in batches_labels:
            noise_level_estimation_df.loc[
                cells_genes_clusters.index.get_level_values(0) == batch,
                noise_level_estimation_df.columns != batch,
            ] = 0

        cells_genes_clusters_equations = pd.concat(
            [observed_umis_df, noise_level_estimation_df, native_expression_df], axis=1
        ).astype(float)

        # Remove equations without information on any gene-metacells cluster pair
        cells_genes_clusters_equations = cells_genes_clusters_equations[
            np.sum(
                cells_genes_clusters_equations.loc[
                    :,
                    cells_genes_clusters_equations.columns.str.startswith("Pgm"),
                ],
                axis=1,
            )
            > 0
        ]

        return cells_genes_clusters_equations

    def _get_valid_batches_and_cells_genes_clusters(
        self,
        cells_genes_clusters: pd.DataFrame,
        min_number_of_pgm_clusters_per_batch: int,
        min_number_of_batches_per_pgm_cluster: int,
    ) -> tuple[pd.Index, pd.Index]:
        """
        Go over all the batches and cells-genes clusters in the current list of pairs and filter out batches or pairs with insufficient data.
        Here we consider insufficient data as a pair with few batches to represent it - which will yield bad estimatoin or, on the other hand, a batch without enough pairs to estimate.
        This process is done by cutting batches and pairs until we get to stagnation.

        :param cells_genes_clusters: The information about the cell genes clusters, all the batches which we've seen those clusters.
        :type cells_genes_clusters: pd.DataFrame

        :param min_number_of_pgm_clusters_per_batch: Batches with less than this amount of different metacell-genes clusters(pgm) won't be calculated due to few data points.
        :type min_number_of_pgm_clusters_per_batch: int

        :param min_number_of_batches_per_pgm_cluster: Metacell-genes clusters(pgm) with less than this amount of different batches won't be calculated due to few data points.
        :type min_number_of_batches_per_pgm_cluster: int

        :return: The names of the batches and metacells-genes clsuters which have enough data for the estimation.
        :rtype: tuple[pd.Index, pd.Index]
        """
        number_of_clusters_changed_in_last_loop = True
        number_of_batches_changed_in_last_loop = True

        valid_cells_genes_clusters = cells_genes_clusters.index.get_level_values(0)
        valid_batches = self.ambient_noise_finder.batches

        cells_genes_clusters_df_reindexed = cells_genes_clusters.reset_index()

        while (
            number_of_clusters_changed_in_last_loop
            or number_of_batches_changed_in_last_loop
        ):

            number_of_batches_percells_genes_clusters = (
                cells_genes_clusters_df_reindexed.groupby(
                    "genes_metacells_cluster"
                ).count()["batch"]
            )

            temp_valid_cells_genes_clusters = number_of_batches_percells_genes_clusters[
                number_of_batches_percells_genes_clusters
                >= min_number_of_batches_per_pgm_cluster
            ].index

            number_of_clusters_changed_in_last_loop = len(
                temp_valid_cells_genes_clusters
            ) != len(valid_cells_genes_clusters)

            valid_cells_genes_clusters = temp_valid_cells_genes_clusters

            # Filter out non-valid genes metacells clusters.
            cells_genes_clusters_df_reindexed = cells_genes_clusters_df_reindexed[
                cells_genes_clusters_df_reindexed["genes_metacells_cluster"].isin(
                    valid_cells_genes_clusters
                )
            ]

            # Remove batches with not enough gmctypes.
            number_of_cells_genes_clusters_per_batch = (
                cells_genes_clusters_df_reindexed.groupby("batch").count()[
                    "genes_metacells_cluster"
                ]
            )
            temp_valid_batches = number_of_cells_genes_clusters_per_batch[
                number_of_cells_genes_clusters_per_batch
                >= min_number_of_pgm_clusters_per_batch
            ].index

            number_of_batches_changed_in_last_loop = len(temp_valid_batches) != len(
                valid_batches
            )
            valid_batches = temp_valid_batches

            cells_genes_clusters_df_reindexed = cells_genes_clusters_df_reindexed[
                cells_genes_clusters_df_reindexed["batch"].isin(valid_batches)
            ]

        return valid_batches, valid_cells_genes_clusters

    def _arrange_and_filter_equations_based_on_valid_batches_and_clusters(
        self,
        noisy_oriented_clusters_equations: pd.DataFrame,
        valid_batches: pd.Index,
        valid_genes_cells_clusters: pd.Index,
    ) -> pd.DataFrame:
        """
        Prepare the equation data frame based on the batches and genes cells clusters by ordering them.
        Filter out equations that do not match the valid batchs or valid pairs

        :param noisy_oriented_clusters_equations: The equations matching the current set of cells-genes for the estimation.
        :type noisy_oriented_clusters_equations: pd.DataFrame

        :param valid_batches: A set of batches whose equations we want to use.
        :type valid_batches: pd.Index

        :param valid_genes_cells_clusters: A set of genes-cells cluster pairs we want to use.
        :type valid_genes_cells_clusters: pd.Index

        :return: The set of equations from before but filtered based on valid batches and valid gene-cells pairs and ordered by the batches and then native expression.
        :rtype: pd.DataFrame
        """
        valid_equations = noisy_oriented_clusters_equations.loc[
            :,
            ["observed"] + list(valid_batches) + list(valid_genes_cells_clusters),
        ]
        valid_equations = valid_equations[
            np.any(valid_equations.loc[:, valid_batches], axis=1)
        ]
        valid_equations = valid_equations[
            np.any(valid_equations.loc[:, valid_genes_cells_clusters], axis=1)
        ]
        return valid_equations

    def _solve_equations(
        self,
        equations: pd.DataFrame,
        number_of_cross_validation_folds: int,
    ) -> pd.DataFrame:
        """
        Take the list of equations and run a solver to get the native expression and noise levels estimation.
        This will be done for different folds based on the number_of_cross_valdiation_folds value.

        :param equations:
                        A representation of the data to be used to estimate the noise.
                        Each row represent one genes-cells cluster pair and batch and hold the information needed to estimate the noise levels and native expression levels.
        :type equations: pd.DataFrame

        :param number_of_cross_validation_folds: The equations will be split to this number of folds, each one of size 90% of the data.
        :type number_of_cross_validation_folds: int

        :return: The mean and std of the coefficients calculated by the solver.
        :rtype: pd.DataFrame
        """
        solver_solutions = []

        # In many cases some columns need to be removed, the solver doesn't do it on it's own so filter out invalid columns.
        valid_columns = [
            "observed"
        ] + self.r_glm_solver.get_valid_column_for_estimation(equations)
        equations = equations.loc[:, valid_columns]

        if number_of_cross_validation_folds == 1:
            solver_solutions.append(
                self.r_glm_solver.fit_coefficents_based_on_equations(equations)
            )

        else:
            for i in range(number_of_cross_validation_folds):
                train, _ = train_test_split(equations, test_size=0.1, random_state=i)
                solver_solutions.append(
                    self.r_glm_solver.fit_coefficents_based_on_equations(train)
                )

        coefficents_values_df = pd.DataFrame(
            {
                "predicted": np.squeeze(np.nanmean(solver_solutions, axis=0)),
                "predicted_sd": np.squeeze(np.nanstd(solver_solutions, axis=0)),
            },
            index=solver_solutions[0].index,
        )

        return coefficents_values_df

    def get_cells_adata_with_noise_level_estimations(
        self,
        estimations_results_obj: NoiseNativeExpressionEstimation,
        estimation_step: int,
    ) -> ad.AnnData:
        """
        Add noise levels estimation to the cells adata, including the noise levels and the umi depth bin for these cells.
        If the cells are above or below a umi depth bin, or no batch estimation was provided for this umi depth bin, we will use the closest estimation available.

        Add `batch_estimated_noise` column to cells obs data which will contain the fraction of noise in this cell based on the batch and estimation.

        :param estimations_results_obj: Holds the full estimation results after the entire ambient noise estimation pipeline.
            This object should have been generated by the same instance of the AmbientNoiseFinder object that is being called.
        :type estimations_results_obj: NoiseNativeExpressionEstimation

        :param estimation_step: The requested step of estimation in estimations_results_obj to use as filler for cells informatiom.
        :type estimation_step: int

        :return: The cell andata which was used in the ambient_noise_finder, now with noise estimation information.
        :rtype: ad.AnnData
        """
        # Add umi depth bin information for cells with too much or too little umis. Using the closest umi depth bin for them.
        cells_adata = self.ambient_noise_finder.cells_adata.copy()
        cells_adata.obs.loc[
            cells_adata.obs.umi_depth
            <= self.ambient_noise_finder.umi_depth_bins_thresholds[0],
            "umi_depth_bin",
        ] = 0

        cells_adata.obs.loc[
            cells_adata.obs.umi_depth
            >= self.ambient_noise_finder.umi_depth_bins_thresholds[-1],
            "umi_depth_bin",
        ] = self.ambient_noise_finder.umi_depth_number_of_bins

        mc.ut.set_o_data(
            cells_adata, "batch_estimated_noise", np.zeros(cells_adata.shape[0])
        )

        for batch_name in cells_adata.obs.batch.unique():
            for umi_depth_bin in range(
                self.ambient_noise_finder.umi_depth_number_of_bins
            ):
                noise_level_estimation = estimations_results_obj.get_noise_estimation_for_batch_by_step_and_umi_depth_bin(
                    batch=batch_name, umi_depth_bin=umi_depth_bin, step=estimation_step
                )

                cells_adata.obs.loc[
                    (cells_adata.obs.batch == batch_name)
                    & (cells_adata.obs.umi_depth_bin == umi_depth_bin),
                    "noise_level_estimation",
                ] = noise_level_estimation

        return cells_adata
