"""
AmbientNoiseEstimator

Object that wrap all the needed functions to estimate the ambient noise levels in a given dataset.
Based on the notion that the observed umis are mixture of the native expressions of the cell state + the noise distribution of a given batch.
Build around the data object `AmbientNoiseFinder` which holds the information about the cells, metacells and batch empty droplets. 
We can estimate both the noise levels and native expressions based on the noise-prone metacells-genes clusters which were discovered using AmbientNoiseFinder.
"""

import functools
import multiprocessing

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split

from AmbientNoiseFinder import AmbientNoiseFinder
from EstimationResults import ContinuousBinEstimation, NoiseNativeExpressionEstimation
from RGlmSolver import RGlmSolver
from utilities import *


class AmbientNoiseEstimator(object):
    def __init__(
        self,
        ambient_noise_finder: AmbientNoiseFinder,
        max_valid_relative_expression_for_pairs: float,
        relative_expression_for_pairs_interval=0.1,
        use_small_metacells_clusters=False,
        use_small_genes_clusters=False,
        max_valid_observed_umis=None,
        min_expected_umi_threshold=100,
    ) -> None:
        """Hold a set of functions to estimate batch ambient noise levels in a dataset in the following flow:

        1. _extract_noisy_oriented_pairs_data: Take the metacells, cells, and ambient noise information from AmbientNoiseFinder and calculate all the relevant cells-genes pairs for noise estimation

        2. estimate_noise_levels_and_native_expression_per_umi_depth_bin:
            - Estimate the noise levels of each umi depth bin by itself, allowing for independent noise levels and native expression for each pair.
            - This will be done for different sets of pairs, each generating a set of equations we can solve.
            - It is up to the user to choose the final set which will be used to the final noise level estimation.

        3. estimate_noise_levels_per_umi_depth_bin_with_shared_native_expression:
            - Estimate the noise levels for each umi depth bin while forcing the same native expression levels across the bins.


        Args:
            ambient_noise_finder (AmbientNoiseFinder): An AmbientNoiseFinder which hold all the information regarding the cells, metacells and empty droplets information from all the batches.

            max_valid_relative_expression_for_pairs (float): Only pairs with expression difference smaller then this will be used to estimate the noise levels and true native expression.
                                                            This allow the user to put an upper limits on cells-genes combinations which are probably no longer dominated by noise.

            relative_expression_for_pairs_interval (float, optional): The addition of more pairs to the set of equations will be govern by this value,
                                                                      each iteration increasing the valid pairs from the minimum to the max_valid_relative_expression_for_pairs by this value.
                                                                      Defaults to 0.1.

            use_small_metacells_clusters (bool, optional): Should we ignore metacells clusters which are marked as small in the AmbientNoiseFinder object. Defaults to False.

            use_small_genes_clusters (bool, optional): Should we ignore genes clusters which are marked as small in the AmbientNoiseFinder object. Defaults to False.

            max_valid_observed_umis (_type_, optional): Should we limit cells with more then specific umi count in the gene modules,
                                                        this can be used to make sure we don't insert outliers cells.
                                                        If no value is given, then we remove cells with more than 4 times the median expresison . Defaults to None.

           min_expected_umi_threshold (int, optional): Provide lower limit for cells-genes pair expected umis given 100% of noise
                                                        anything lower then this won't allow to get a precise estimation. Defaults to 100.
        """
        self.ambient_noise_finder = ambient_noise_finder
        self.max_valid_relative_expression_for_pairs = (
            max_valid_relative_expression_for_pairs
        )

        self.relative_expression_for_pairs_interval = (
            relative_expression_for_pairs_interval
        )

        # This template mostly saves running time by doing some calculations and assignments one time.
        self.cells_genes_clusters_pair_template = (
            self._create_cells_genes_clusters_pair_template()
        )

        self.cells_genes_clusters_pairs = self._extract_noisy_oriented_pairs_data(
            use_small_metacells_clusters=use_small_metacells_clusters,
            use_small_genes_clusters=use_small_genes_clusters,
            max_valid_observed_umis=max_valid_observed_umis,
            min_expected_umi_threshold=min_expected_umi_threshold,
        )

        # Initialized the a R based GLM solver to handle and solve the equations.
        self.r_glm_solver = RGlmSolver()

    def estimate_noise_levels_and_native_expression_per_umi_depth_bin(
        self,
        min_number_of_pairs_per_batch=10,
        min_number_of_batches_per_pair=5,
        number_of_processes=-1,
        number_of_cross_validation_folds=10,
    ) -> dict[int:ContinuousBinEstimation]:
        """
        - Estimate the noise levels of each umi depth bin by itself, allowing for independent noise levels and native expression for each pair.
        - This will be done for different sets of pairs, each generating a set of equations we can solve.
        - It is up to the user to choose the final set which will be used to the final noise level estimation.

        Args:
            min_number_of_pairs_per_batch (int, optional): Prevent estimation with low amount of pairs per batch to guarantee a precise estimation of noise levels. Defaults to 10.

            min_number_of_batches_per_pair (int, optional): Prevent estimation with low amount of batch per pair to guarantee a precise estimation of native expression. Defaults to 5.

            number_of_processes (int, optional): Allow to run each umi depth bin in a different process, the value which will be used is max(# umi depth bins, number_of_processes). -1 means no multiprocess. Defaults to -1.

            number_of_cross_validation_folds (int, optional): Estimate the noise levels and true native experssion this number of times on different folds of the equations. Defaults to 10.

        Returns:
            dict[int: ContinuousBinEstimation]: The results of the estimation, one for each umi depth bin.
                                                This will hold the continues information for the entire of equations from the smallest one to the largest one.
        """
        number_of_processes = get_number_of_valid_processes(number_of_processes)
        number_of_processes = min(
            number_of_processes, self.ambient_noise_finder.umi_depth_number_of_bins
        )

        if number_of_processes > 1:
            with multiprocessing.Pool(number_of_processes) as multiprocess_pool:
                estimations_results = multiprocess_pool.map(
                    functools.partial(
                        self._estimate_noise_levels_and_native_expression_single_bin,
                        **locals()
                    ),
                    range(1, self.ambient_noise_finder.umi_depth_number_of_bins + 1),
                )

        else:
            estimations_results = [
                self._estimate_noise_levels_and_native_expression_single_bin(
                    umi_depth_bin,
                    min_number_of_pairs_per_batch=min_number_of_pairs_per_batch,
                    min_number_of_batches_per_pair=min_number_of_batches_per_pair,
                    number_of_cross_validation_folds=number_of_cross_validation_folds,
                )
                for umi_depth_bin in range(
                    1, self.ambient_noise_finder.umi_depth_number_of_bins + 1
                )
            ]

        estimations_results = {
            bin_estimation.bin_index: bin_estimation
            for bin_estimation in estimations_results
        }

        return estimations_results

    def estimate_noise_levels_per_umi_depth_bin_with_shared_native_expression(
        self,
        estimations_results_per_bin: dict[int:ContinuousBinEstimation],
        relative_expression_for_pairs: float,
        min_number_of_pairs_per_batch=10,
        min_number_of_batches_per_pair=5,
        number_of_cv=10,
    ) -> NoiseNativeExpressionEstimation:
        """Estimate the noise levels for each umi depth bin while forcing the same native expression levels across the bins.

        Args:
            estimations_results_per_bin (dict[int: ContinuousBinEstimation]): The results of the function estimate_noise_levels_and_native_expression_per_umi_depth_bin.
                                                                              This is the results of the continues estimation, one for each umi depth bin.

            relative_expression_for_pairs (float): Only pairs with expression difference smaller then this will be used to estimate the noise levels.

            min_number_of_pairs_per_batch (int, optional): Batches with smaller number of cells-genes pairs to represent them will be removed.
            This prevent estimation with low amount of pairs per batch to guarantee a precise estimation of noise levels. Defaults to 10.

            min_number_of_batches_per_pair (int, optional): Cells-gense pairs with smaller number of batches to represent them will be removed
            This prevent estimation with low amount of batch per pair to guarantee a precise estimation of native expression. Defaults to 5.

            number_of_cross_validation_folds (int, optional): Estimate the noise levels and true native experssion this number of times on different folds of the equations. Defaults to 10.

        Returns:
            NoiseNativeExpressionEstimation: Holds the noise levels estimation of all the batches across the different umi depth bins.
                                             Also holds the estimated native expression which is shared across the different umi depth bins.
        """
        # Extract the genes-metacells pairs which are shared across different umi depth bins - only those pairs might be use.
        common_genes_metacells_clusters_pairs_between_bins = (
            self._get_common_genes_metacells_clusters_pairs_between_bins(
                estimations_results_per_bin, relative_expression_for_pairs
            )
        )

        # Make sure that the provided relative_expression_for_pairs holds enough information based on the provided min_number_of_pairs_per_batch.
        assert (
            len(common_genes_metacells_clusters_pairs_between_bins)
            > min_number_of_pairs_per_batch
        ), "The number of shared genes metacells clusters pairs < min_number_of_pairs_per_batch, can't continue"

        umi_depth_equations_list = []
        for umi_depth_bin in range(
            1, self.ambient_noise_finder.umi_depth_number_of_bins + 1
        ):
            current_bin_cells_genes_clusters_pairs = (
                self.cells_genes_clusters_pairs.loc[umi_depth_bin]
            )
            cells_genes_clusters_pairs_equations = (
                self._format_cells_genes_clusters_pairs_dataframe_as_equations(
                    current_bin_cells_genes_clusters_pairs
                )
            )

            # Remove equations without information on any gene-metacells cluster pair
            equations_rows_genes_metacells_clusters_pairs_label = (
                cells_genes_clusters_pairs_equations.columns[
                    cells_genes_clusters_pairs_equations.columns.str.startswith("Pgm")
                ][
                    np.where(
                        cells_genes_clusters_pairs_equations.loc[
                            :,
                            cells_genes_clusters_pairs_equations.columns.str.startswith(
                                "Pgm"
                            ),
                        ]
                        > 0
                    )[1]
                ]
            )

            index_of_equations_with_shared_native_expression = np.isin(
                equations_rows_genes_metacells_clusters_pairs_label,
                common_genes_metacells_clusters_pairs_between_bins,
            )

            valid_equations_of_shared_native_expression = (
                cells_genes_clusters_pairs_equations[
                    index_of_equations_with_shared_native_expression
                ]
            )

            (
                valid_batches,
                valid_cells_genes_clusters_pairs,
            ) = self._get_valid_batches_and_cells_genes_clusters_pairs(
                cells_genes_clusters_pairs=current_bin_cells_genes_clusters_pairs,
                min_number_of_pairs_per_batch=min_number_of_pairs_per_batch,
                min_number_of_batches_per_pairs=min_number_of_batches_per_pair,
            )

            valid_equations = (
                self._arrange_and_filter_equations_based_on_valid_batches_and_pairs(
                    noisy_oriented_pairs_equations=valid_equations_of_shared_native_expression,
                    valid_batches=valid_batches,
                    valid_genes_cells_clusters_pairs=valid_cells_genes_clusters_pairs,
                )
            )

            # Add label to the different batches based on the umi depth bins they represent. This will allow different estimation for different bins but the same for the genes-cells pairs.
            columns_labels_with_bin_reference = [
                "%s_umi_depth_bin_%s" % (column, umi_depth_bin)
                if column in valid_batches
                else column
                for column in valid_equations.columns
            ]
            valid_equations.columns = columns_labels_with_bin_reference
            umi_depth_equations_list.append(valid_equations)

        # Combine all the equations from the different umi depth bins
        all_umi_depth_equations = pd.concat(umi_depth_equations_list)
        all_umi_depth_equations = all_umi_depth_equations.fillna(0)

        (
            batches_noise_estimation_df,
            cells_genes_pair_native_expression_estimation_df,
        ) = self._solve_equations(
            all_umi_depth_equations,
            number_of_cross_validation_folds=number_of_cv,
            max_relative_expression_for_pairs=relative_expression_for_pairs,
        )

        return NoiseNativeExpressionEstimation(
            batches_noise_estimation_df,
            cells_genes_pair_native_expression_estimation_df,
            valid_equations,
            relative_expression_for_pairs,
        )

    def _get_common_genes_metacells_clusters_pairs_between_bins(
        self,
        estimations_results_per_bin: dict[int:ContinuousBinEstimation],
        relative_expression_for_pairs: float,
    ) -> pd.Index:
        """Get the set of genes metacells clusters pairs (Pgm) which is shared between different umi depth bins.

        Args:
            estimations_results_per_bin (dict[int: ContinuousBinEstimation]): The results of the function estimate_noise_levels_and_native_expression_per_umi_depth_bin.
                                                                              This is the results of the continues estimation, one for each umi depth bin.

            relative_expression_for_pairs (float): Only pairs with expression difference smaller then this will be used to estimate the noise levels.

        Returns:
            pd.Index: A set of all the shared genes metacells clusters pairs between differet umi depth bins
        """
        valid_genes_metacells_clusters_pairs = None
        for bin_index in estimations_results_per_bin:
            bin_valid_genes_metacells_clusters_pairs = estimations_results_per_bin[
                bin_index
            ].get_valid_genes_metacells_clusters_pairs_for_relative_expression_value(
                relative_expression_for_pairs
            )

            if valid_genes_metacells_clusters_pairs is None:
                valid_genes_metacells_clusters_pairs = (
                    bin_valid_genes_metacells_clusters_pairs
                )
            else:
                valid_genes_metacells_clusters_pairs = (
                    valid_genes_metacells_clusters_pairs
                    & bin_valid_genes_metacells_clusters_pairs
                )

        return valid_genes_metacells_clusters_pairs

    def _create_cells_genes_clusters_pair_template(self):
        """Generate and calculate all the information needed for the cells-genes clusters pairs.
        This include information about the total umis of cells, the batch empty droplets information and other information which run faster if calculated once and then bein copy vs
         generating the DataFrame everytime.

        Returns:
            pd.DataFrame: Hold all the relevent information for each genes-metacells clusters, and already build to be filled by specific information for each pair.
            This will later be used to actually generate the equations for the solver based on the observed umis, the total umis and the empty droplets fractions information.
        """
        cells_genes_cluster_pair_template = pd.DataFrame(
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

        cells_genes_cluster_pair_template[
            "metacell"
        ] = self.ambient_noise_finder.cells_adata.obs.loc[
            cells_genes_cluster_pair_template.index
        ].metacell

        cells_genes_cluster_pair_template[
            "total_umis"
        ] = self.ambient_noise_finder.cells_adata.obs.umi_depth

        cells_genes_cluster_pair_template[
            "batch"
        ] = self.ambient_noise_finder.cells_adata.obs.loc[
            cells_genes_cluster_pair_template.index
        ].batch

        cells_genes_cluster_pair_template[
            "cells_cluster"
        ] = self.ambient_noise_finder.cells_adata.obs.loc[
            cells_genes_cluster_pair_template.index
        ].cells_cluster

        cells_genes_cluster_pair_template[
            "umi_depth_bin"
        ] = self.ambient_noise_finder.cells_adata.obs.loc[
            cells_genes_cluster_pair_template.index
        ].umi_depth_bin

        return cells_genes_cluster_pair_template

    def _extract_noisy_oriented_pairs_data(
        self,
        use_small_metacells_clusters: bool,
        use_small_genes_clusters: bool,
        max_valid_observed_umis: int,
        min_expected_umi_threshold: int,
    ) -> pd.DataFrame:
        """
        Go over all the valid metacells-genes clusters pairs which are deemed to be noisy oriented (below the max_valid_relative_expression_for_pairs) and collect all the Pgm
        information needed for later estimations. For each pair, split the data based on batch and umi depth bin.

        Args:
            use_small_metacells_clusters (bool): Should we ignore metacells clusters which are marked as small in the AmbientNoiseFinder object.

            use_small_genes_clusters (bool): Should we ignore genes clusters which are marked as small in the AmbientNoiseFinder object.

            max_valid_observed_umis (int): Should we limit cells with more then specific umi count in the gene modules,
                                            this can be used to make sure we don't insert outliers cells.
                                            If no value is given, then we remove cells with more than 4 times the median expresison.


            min_expected_umi_threshold (int): Provide lower limit for cells-genes pair expected umis given 100% of noise
                                              anything lower then this won't allow to get a precise estimation.

        Returns:
            pd.DataFrame: Holds one line for each combination of genes-metacells cluster pair, batch, umi depth bins. Filtered based on user definition.
        """
        cells_genes_clusters_pairs = []
        already_calculated_pairs = []
        number_of_digits = max(
            ("%s" % self.relative_expression_for_pairs_interval)[::-1].find("."), 0
        )

        metacells_genes_pair_relative_expression_df = (
            self.ambient_noise_finder.metacells_genes_pair_relative_expression_df
        )

        for current_relative_expression in np.arange(
            metacells_genes_pair_relative_expression_df.min().min(),
            self.max_valid_relative_expression_for_pairs
            + self.relative_expression_for_pairs_interval,
            self.relative_expression_for_pairs_interval,
        ):
            current_relative_expression = float(
                ("{:.%df}" % number_of_digits).format(current_relative_expression)
            )

            combined_metacells_genes_cluster_pair_noisy_information = pd.DataFrame()

            # Get the set of valid metacells clusters and genes clusters
            metacell_cluster_list, gene_cluster_list = np.where(
                metacells_genes_pair_relative_expression_df
                <= current_relative_expression
            )

            for j, metacells_cluster in enumerate(metacell_cluster_list):
                genes_cluster = metacells_genes_pair_relative_expression_df.columns[
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
                if label in already_calculated_pairs:
                    continue

                already_calculated_pairs.append(label)

                # Extract the specific pair information
                metacells_genes_cluster_pair_noisy_information = (
                    self._get_metacells_genes_cluster_pair_noisy_information(
                        metacells_cluster, genes_cluster
                    )
                )

                # Defien the upper limit of observed umis per cells, either by the user definition or from the data itself.
                max_valid_observed_umis = (
                    (
                        metacells_genes_cluster_pair_noisy_information.observed.median()
                        + 1
                    )
                    * 4
                    if max_valid_observed_umis is None
                    else max_valid_observed_umis
                )

                metacells_genes_cluster_pair_noisy_information = (
                    metacells_genes_cluster_pair_noisy_information[
                        metacells_genes_cluster_pair_noisy_information.observed
                        <= max_valid_observed_umis
                    ]
                )

                # Combine the information from all the metacell-genes cluster pair based on origin batch and umi depth bin
                combined_metacells_genes_cluster_pair_noisy_information = (
                    metacells_genes_cluster_pair_noisy_information.groupby(
                        ["umi_depth_bin", "batch"]
                    ).agg({"observed": "sum", "expected": "sum", "total_umis": "sum"})
                )

                # Filter pairs without observed umis or with small number of expected umis
                combined_metacells_genes_cluster_pair_noisy_information = combined_metacells_genes_cluster_pair_noisy_information[
                    (
                        combined_metacells_genes_cluster_pair_noisy_information.expected
                        >= min_expected_umi_threshold
                    )
                    & (
                        combined_metacells_genes_cluster_pair_noisy_information.observed
                        > 0
                    )
                ]

                # Add relevant labels to identify this pair
                combined_metacells_genes_cluster_pair_noisy_information[
                    "relative_expression"
                ] = current_relative_expression

                combined_metacells_genes_cluster_pair_noisy_information[
                    "genes_metacells_pair"
                ] = "Pgm_%s_in_%s" % (genes_cluster, metacells_cluster)

                combined_metacells_genes_cluster_pair_noisy_information = (
                    combined_metacells_genes_cluster_pair_noisy_information.set_index(
                        "genes_metacells_pair", append=True
                    )
                )

                combined_metacells_genes_cluster_pair_noisy_information.reorder_levels(
                    ["umi_depth_bin", "genes_metacells_pair", "batch"]
                )

            cells_genes_clusters_pairs.append(
                combined_metacells_genes_cluster_pair_noisy_information
            )

        return pd.concat(cells_genes_clusters_pairs)

    def _get_metacells_genes_cluster_pair_noisy_information(
        self, metacells_cluster: int, genes_cluster: int
    ) -> pd.DataFrame:
        """Fill in specific metacells-genes cluster pair information in a cells_genes_clusters_pair template.

        Args:
            metacells_cluster (int): The id of the metacells cluster.

            genes_cluster (int): The id of the genes cluster.

        Returns:
            pd.DataFrame: Hold all the relevent information for a specific genes-metacells cluster.
            This will later be used to actually generate the equations for the solver based on the observed umis, the total umis and the empty droplets fractions information.
        """

        pair_metacells = self.ambient_noise_finder.metacells_adata.obs[
            self.ambient_noise_finder.metacells_adata.obs.metacells_cluster
            == metacells_cluster
        ].index

        pair_genes = self.ambient_noise_finder.metacells_adata.var[
            self.ambient_noise_finder.metacells_adata.var.genes_cluster == genes_cluster
        ].index

        cells_genes_clusters_pair = self.cells_genes_clusters_pair_template.copy()
        cells_genes_clusters_pair = cells_genes_clusters_pair[
            np.in1d(cells_genes_clusters_pair.metacell, pair_metacells.astype(int))
        ]

        # Add the specific empty droplets fraction information based on the batch and the genes cluster we work with.
        for (
            batch
        ) in self.ambient_noise_finder.empty_droplet_genes_cluster_fraction.index:
            cells_genes_clusters_pair.loc[
                cells_genes_clusters_pair.batch == batch, "empty_droplets_fraction"
            ] = self.ambient_noise_finder.empty_droplet_genes_cluster_fraction.loc[
                batch, genes_cluster
            ]

        cells_genes_clusters_pair["observed"] = np.squeeze(
            np.asarray(
                self.ambient_noise_finder.cells_adata[
                    cells_genes_clusters_pair.index, pair_genes
                ].X.sum(axis=1)
            )
        )
        cells_genes_clusters_pair["expected"] = (
            cells_genes_clusters_pair["empty_droplets_fraction"]
            * cells_genes_clusters_pair["total_umis"]
        )

        return cells_genes_clusters_pair

    def _estimate_noise_levels_and_native_expression_single_bin(
        self,
        current_bin_to_estimate: int,
        min_number_of_pairs_per_batch: int,
        min_number_of_batches_per_pair: int,
        number_of_cross_validation_folds: int,
    ) -> ContinuousBinEstimation:
        """Perform estimation over a specific umi depth bin independently of other bins.


        Args:
            current_bin_to_estimate (int): id of the estimated umi depth bin to estimate
            min_number_of_pairs_per_batch (int): Remove batch with smaller number of genes-metacells pairs representing them.
            This prevent estimation with low amount of pairs per batch to guarantee a precise estimation of noise levels.

            min_number_of_batches_per_pair (int):  Remove pairs with smaller number of batches representing them.
            This prevent estimation with low amount of batch per pair to guarantee a precise estimation of native expression.

            number_of_cross_validation_folds (int): Estimate the noise levels and true native experssion this number of times on different folds of the equations.

        Returns:
            ContinuousBinEstimation: Hold the noise levels and native expression estimation of all batches and cells-genes pairs for this specific bin.
            This object also hold the estimation per each set of pairs being added to the estimation, so per different relative expression of pairs
        """
        number_of_digits = max(
            ("%s" % self.relative_expression_for_pairs_interval)[::-1].find("."), 0
        )

        batches_noise_estimations_list = []
        cells_genes_pair_native_experssion_estimation_list = []
        valid_pairs_tracker = []
        valid_batches_tracker = []
        already_used_cells_genes_pairs = []
        already_covered_batches = []

        current_bin_cells_genes_clusters_pairs = self.cells_genes_clusters_pairs.loc[
            current_bin_to_estimate
        ]

        cells_genes_clusters_pairs_equations = (
            self._format_cells_genes_clusters_pairs_dataframe_as_equations(
                current_bin_cells_genes_clusters_pairs
            )
        )

        # Go over all the valid relative expresisons from the minimum to the maximum and add more and more pairs to the estimation. This will add more and more equations.
        for current_max_relative_expression_for_pairs in tqdm.tqdm(
            np.arange(
                current_bin_cells_genes_clusters_pairs.relative_expression.min(),
                self.max_valid_relative_expression_for_pairs
                + self.relative_expression_for_pairs_interval,
                self.relative_expression_for_pairs_interval,
            ),
            desc="Estimating noise levels and true distribution for bin size: %s"
            % current_bin_to_estimate,
        ):

            current_max_relative_expression_for_pairs = float(
                ("{:.%df}" % number_of_digits).format(
                    current_max_relative_expression_for_pairs
                )
            )
            cells_genes_clusters_pairs_for_current_max_relative_expression = (
                current_bin_cells_genes_clusters_pairs[
                    current_bin_cells_genes_clusters_pairs.relative_expression
                    <= current_max_relative_expression_for_pairs
                ]
            )

            # If no new pairs were added continue to the next relative_expression value
            if len(
                cells_genes_clusters_pairs_for_current_max_relative_expression
            ) == len(already_used_cells_genes_pairs):
                valid_batches_tracker.append(len(already_covered_batches))
                valid_pairs_tracker.append(len(already_used_cells_genes_pairs))
                continue

            (
                valid_batches,
                valid_cells_genes_clusters_pairs,
            ) = self._get_valid_batches_and_cells_genes_clusters_pairs(
                cells_genes_clusters_pairs=cells_genes_clusters_pairs_for_current_max_relative_expression,
                min_number_of_pairs_per_batch=min_number_of_pairs_per_batch,
                min_number_of_batches_per_pairs=min_number_of_batches_per_pair,
            )

            if len(valid_batches) == 0 or len(valid_cells_genes_clusters_pairs) == 0:
                valid_batches_tracker.append(0)
                valid_pairs_tracker.append(0)
                continue

            else:
                valid_pairs_tracker.append(len(valid_cells_genes_clusters_pairs))
                valid_batches_tracker.append(len(valid_batches))

            valid_equations = (
                self._arrange_and_filter_equations_based_on_valid_batches_and_pairs(
                    noisy_oriented_pairs_equations=cells_genes_clusters_pairs_equations,
                    valid_batches=valid_batches,
                    valid_genes_cells_clusters_pairs=valid_cells_genes_clusters_pairs,
                )
            )

            (
                batches_noise_estimation_df,
                cells_genes_pair_native_expression_estimation_df,
            ) = self._solve_equations(
                valid_equations,
                number_of_cross_validation_folds=number_of_cross_validation_folds,
                max_relative_expression_for_pairs=current_max_relative_expression_for_pairs,
            )

            batches_noise_estimations_list.append(batches_noise_estimation_df)
            cells_genes_pair_native_experssion_estimation_list.append(
                cells_genes_pair_native_expression_estimation_df
            )

        bin_estimation = ContinuousBinEstimation(
            umi_depth_bin_index=current_bin_to_estimate,
            batches_noise_estimation_list=batches_noise_estimations_list,
            cells_genes_pair_native_experssion_estimation_list=cells_genes_pair_native_experssion_estimation_list,
            min_valid_relative_expression_for_pairs=current_bin_cells_genes_clusters_pairs.relative_expression.min(),
            max_valid_relative_expression_for_pairs=self.max_valid_relative_expression_for_pairs
            + self.relative_expression_for_pairs_interval,
        )

        return bin_estimation

    def _format_cells_genes_clusters_pairs_dataframe_as_equations(
        self, cells_genes_clusters_pairs: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert a dataframe that holds all the data about a specific genes-cells cluster pair, batch and umi depth to an equation format dataframe ready for estimation.
        This is mostly rearranging the data differently to represent the equation:

            observeed = (batch_noise_levels * empty_droplets_fraction + native_expression) * total_umis

        Args:
            cells_genes_clusters_pairs (pd.DataFrame): A df that holds one line for each combination of genes-metacells cluster pair, batch, umi depth bins.

        Returns:
            pd.DataFrame: An equation based dataframe to represent the noisy data in a solveable manner for the next steps.
            Each row is a different equation based on cells-genes cluster, batch and umi depth.
        """
        number_of_equations = cells_genes_clusters_pairs.shape[0]
        number_of_cells_genes_clusters_pairs = len(
            cells_genes_clusters_pairs.index.get_level_values(1).unique()
        )

        number_of_batches = len(
            cells_genes_clusters_pairs.index.get_level_values(0).unique()
        )

        observed_umis_df = pd.DataFrame(
            cells_genes_clusters_pairs["observed"].values, columns=["observed"]
        )

        # Generate a dataframe with a repeated vector of native expression, in each row we have the native expression coefficent (total umis) of this specific cell-genes pairs
        native_expression_df = pd.DataFrame(
            np.repeat(
                cells_genes_clusters_pairs.total_umis.values,
                number_of_cells_genes_clusters_pairs,
            ).reshape(number_of_equations, number_of_cells_genes_clusters_pairs),
            columns=cells_genes_clusters_pairs.index.get_level_values(1).unique(),
        )
        # Go over all the rows(cells_genes_clusters_pair), remove value from true native columns which doesn't match the row pair
        for (
            cells_genes_clusters_pair
        ) in cells_genes_clusters_pairs.index.get_level_values(1).unique():
            native_expression_df.loc[
                cells_genes_clusters_pairs.index.get_level_values(1)
                == cells_genes_clusters_pair,
                native_expression_df.columns != cells_genes_clusters_pair,
            ] = 0

        # Generate a dataframe with a repeated vector of noise levels estimation, in each row we have the noise levels coefficent (total umis * empty droplets fraction) of this specific cell-genes pairs
        noise_level_estimation_df = pd.DataFrame(
            np.repeat(
                cells_genes_clusters_pairs.expected.values, number_of_batches
            ).reshape(number_of_equations, number_of_batches),
            columns=cells_genes_clusters_pairs.index.get_level_values(0).unique(),
        )

        # Go over all the rows(batch), remove value from batch columns which doesn't match the row batch
        for batch in cells_genes_clusters_pairs.index.get_level_values(0).unique():
            noise_level_estimation_df.loc[
                cells_genes_clusters_pairs.index.get_level_values(0) == batch,
                noise_level_estimation_df.columns != batch,
            ] = 0

        return pd.concat(
            [observed_umis_df, noise_level_estimation_df, native_expression_df], axis=1
        ).astype(np.float64)

    def _get_valid_batches_and_cells_genes_clusters_pairs(
        self,
        cells_genes_clusters_pairs: pd.DataFrame,
        min_number_of_pairs_per_batch: int,
        min_number_of_batches_per_pairs: int,
    ) -> tuple[pd.Index, pd.Index]:
        """Go over all the batches and cells-genes clusters pairs in the current list of pairs and filter out batchs or pairs with insufficient data.
        Here we consider insufficent data as a pair with not enough batches to represent it - which will yield bad estimatoin or on the other hand batch without enough pairs to estimate on.
        This process is being done by cutting batches and pairs until we get to stagnation.

        Args:
            cells_genes_clusters_pairs (pd.DataFrame): A df that holds one line for each combination of genes-metacells cluster pair, batch, umi depth bins.

            min_number_of_pairs_per_batch (int):  Filter out batch with smaller number of genes-metacells pairs representing them.
            This prevent estimation with low amount of pairs per batch to guarantee a precise estimation of noise levels.

            min_number_of_batches_per_pairs (int): Filter out pairs with smaller number of batches representing them.
            This prevent estimation with low amount of batch per pair to guarantee a precise estimation of native expression.

        Returns:
            tuple[pd.Index, pd.Index]: Two sets, one represent the valid batches and the other the valid cells_genes_clusters_pairs.
        """

        number_of_pairs_changed_in_last_loop, number_of_batches_changed_in_last_loop = (
            True,
            True,
        )

        valid_cells_genes_clusters_pairs = (
            cells_genes_clusters_pairs.index.get_level_values(0)
        )

        valid_batches = self.ambient_noise_finder.batches

        cells_genes_clusters_pairs_df_reindexed = (
            cells_genes_clusters_pairs.reset_index()
        )

        while (
            number_of_pairs_changed_in_last_loop
            or number_of_batches_changed_in_last_loop
        ):
            number_of_batches_percells_genes_clusters_pairs = (
                cells_genes_clusters_pairs_df_reindexed.groupby(
                    "genes_metacells_pair"
                ).count()["batch"]
            )

            temp_valid_cells_genes_clusters_pairs = (
                number_of_batches_percells_genes_clusters_pairs[
                    number_of_batches_percells_genes_clusters_pairs
                    >= min_number_of_batches_per_pairs
                ].index
            )
            number_of_pairs_changed_in_last_loop = len(
                temp_valid_cells_genes_clusters_pairs
            ) != len(valid_cells_genes_clusters_pairs)

            valid_cells_genes_clusters_pairs = temp_valid_cells_genes_clusters_pairs

            # Filter out non-valid genes metacells clusters.
            cells_genes_clusters_pairs_df_reindexed = (
                cells_genes_clusters_pairs_df_reindexed[
                    cells_genes_clusters_pairs_df_reindexed[
                        "genes_metacells_pair"
                    ].isin(valid_cells_genes_clusters_pairs)
                ]
            )

            # Remove batches with not enough gmctypes.
            number_of_cells_genes_clusters_pairs_per_batch = (
                cells_genes_clusters_pairs_df_reindexed.groupby("batch").count()[
                    "genes_metacells_pair"
                ]
            )
            temp_valid_batches = number_of_cells_genes_clusters_pairs_per_batch[
                number_of_cells_genes_clusters_pairs_per_batch
                >= min_number_of_pairs_per_batch
            ].index

            number_of_batches_changed_in_last_loop = len(temp_valid_batches) != len(
                valid_batches
            )
            valid_batches = temp_valid_batches

            cells_genes_clusters_pairs_df_reindexed = (
                cells_genes_clusters_pairs_df_reindexed[
                    cells_genes_clusters_pairs_df_reindexed["batch"].isin(valid_batches)
                ]
            )

        return valid_batches, valid_cells_genes_clusters_pairs

    def _arrange_and_filter_equations_based_on_valid_batches_and_pairs(
        self,
        noisy_oriented_pairs_equations: pd.DataFrame,
        valid_batches: pd.Index,
        valid_genes_cells_clusters_pairs: pd.Index,
    ) -> pd.DataFrame:
        """
        Prepare the equation dataframe based on the batches and genes cells clusters pairs by reording them.
        Also filter out equations which don't match the valid batchs or valid pairs

        Args:
            noisy_oriented_pairs_equations (pd.DataFrame): A representation of the data to be used to estimate the noise.
            Each row represent one genes-cells cluster pair and batch and hold the information needed to estimate the noise levels and native expression levels.

            valid_batches (pd.Index): A set of batches whose equations we want to use.

            valid_pairs (pd.Index): A set of genes-cells cluster pairs we want to use.

        Returns:
            pd.DataFrame: The set of equations from before but filtered based on valid batches and valid genes-cells pairs and ordered by the batchs and then expression pair.
        """
        valid_equations = noisy_oriented_pairs_equations.loc[
            :,
            ["observed"] + list(valid_batches) + list(valid_genes_cells_clusters_pairs),
        ]
        valid_equations = valid_equations[
            np.any(valid_equations.loc[:, valid_batches], axis=1)
        ]
        valid_equations = valid_equations[
            np.any(valid_equations.loc[:, valid_genes_cells_clusters_pairs], axis=1)
        ]
        return valid_equations

    def _solve_equations(
        self,
        noisy_oriented_pairs_equations: pd.DataFrame,
        number_of_cross_validation_folds: int,
        max_relative_expression_for_pairs: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Take the list of equations and run a solver to get the native expression and noise levels estimation.
        This will be done for different folds based on the number_of_cross_valdiation_folds value.

        Args:
            noisy_oriented_pairs_equations (pd.DataFrame): A representation of the data to be used to estimate the noise.
            Each row represent one genes-cells cluster pair and batch and hold the information needed to estimate the noise levels and native expression levels.

            number_of_cross_validation_folds (int): Estimate the noise levels and true native experssion this number of times on different folds of the equations.

            max_relative_expression_for_pairs (float): The maximum value of relative expressions which was used to generate those equations set, this will alter be used to find the optimal cuttoff by the user.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Two dataframes with the estimations results, the first for the noise levels and the second for the native expression levels/
        """
        solver_solutions = []

        # In many cases some columns need to be removed, the solver doesn't do it on it's own so filter out invalid columns.
        valid_columns = [
            "observed"
        ] + self.r_glm_solver.get_valid_column_for_estimation(
            noisy_oriented_pairs_equations
        )
        noisy_oriented_pairs_equations = noisy_oriented_pairs_equations.loc[
            :, valid_columns
        ]

        if number_of_cross_validation_folds == 1:
            solver_solutions.append(
                self.r_glm_solver.fit_coefficents_based_on_equations(
                    noisy_oriented_pairs_equations
                )
            )

        else:
            for i in range(number_of_cross_validation_folds):
                train, _ = train_test_split(
                    noisy_oriented_pairs_equations, test_size=0.1, random_state=i
                )
                solver_solutions.append(
                    self.r_glm_solver.fit_coefficents_based_on_equations(train)
                )

        coefficents_values_df = pd.DataFrame(
            {
                "predicted": np.squeeze(np.nanmean(solver_solutions, axis=0)),
                "predicted_sd": np.squeeze(np.nanstd(solver_solutions, axis=0)),
                "max_relative_expression_for_pairs": max_relative_expression_for_pairs,
            },
            index=solver_solutions[0].index,
        )

        # Split the results to two dataframes, one for batches and another for native expression.
        batches_noise_estimation_df = coefficents_values_df[
            ~coefficents_values_df.index.str.startswith("Pgm")
        ]
        batches_noise_estimation_df["predicted_%"] = (
            batches_noise_estimation_df["predicted"] * 100
        )
        batches_noise_estimation_df["predicted_sd_%"] = (
            batches_noise_estimation_df["predicted_sd"] * 100
        )

        if "bin" in batches_noise_estimation_df.columns:
            batches_noise_estimation_df.bin = batches_noise_estimation_df.bin.astype(
                np.int
            )

        cells_genes_pair_native_expression_estimation_df = coefficents_values_df[
            coefficents_values_df.index.str.startswith("Pgm")
        ]

        if "bin" in cells_genes_pair_native_expression_estimation_df.columns:
            cells_genes_pair_native_expression_estimation_df = (
                cells_genes_pair_native_expression_estimation_df.drop("bin", axis=1)
            )

        return (
            batches_noise_estimation_df,
            cells_genes_pair_native_expression_estimation_df,
        )
