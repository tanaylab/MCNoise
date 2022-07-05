"""
Two objects to hold the information from the estimation process. 
The first one is ContinuousBinEstimation - this is estimation per umi depth bin throughout the different relative expression values the user provided.
The second one is NoiseNativeExpressionEstimation - this holds the final estimation for batch noise levels per umi depth with the shared native expressions values.

Each object holds different sets of functions usefull to plotting of the data to help the user understand the performence of the estimation process.

"""

import pandas as pd


class ContinuousBinEstimation(object):
    def __init__(
        self,
        umi_depth_bin_index: int,
        batches_noise_estimation_list: list[pd.DataFrame],
        cells_genes_pair_native_experssion_estimation_list: list[pd.DataFrame],
        min_valid_relative_expression_for_pairs: float,
        max_valid_relative_expression_for_pairs: float,
    ) -> None:
        """Hold all the information for a specific umi depth bin estimation.
        This includes the estimation using various sets of genes-cells pairs which is defined by the relative epxression of those pairs.

        Args:
            umi_depth_bin_index (int): The index of the umi depth bin.

            batches_noise_estimation_list (list[pd.DataFrame]): All the noise levels estimation across different sets of cells-genes clusters pairs.

            cells_genes_pair_native_experssion_estimation_list (list[pd.DataFrame]):  All the native expression estimation across different sets of cells-genes clusters pairs.

            min_valid_relative_expression_for_pairs (float): The smallest possible relative expression for the pairs used in estimation.
            This is based on the data itself and should represent those pairs which are more likely originated only from noise.

            max_valid_relative_expression_for_pairs (float): The largest relative expression for the pairs used in estimation.
            This is defined by the user and represent the top limit of pairs which can still be considered mostly noise oriented.
        """

        self.bin_index = umi_depth_bin_index

        self.batches_noise_estimation = pd.concat(batches_noise_estimation_list)

        self.cells_genes_pair_native_experssion_estimation = pd.concat(
            cells_genes_pair_native_experssion_estimation_list
        )

        self.min_valid_relative_expression_for_pairs = (
            min_valid_relative_expression_for_pairs
        )

        self.max_valid_relative_expression_for_pairs = (
            max_valid_relative_expression_for_pairs
        )

    def get_valid_genes_metacells_clusters_pairs_for_relative_expression_value(
        self, relative_expression_value: float
    ) -> pd.Index:
        """Extract all the pairs which were used for estimation up until a specific relative expression value.

        Args:
            relative_expression_value (float): The maximum relative expression value which we allow to be used in estimation.

        Returns:
            pd.Index: The names of all the pairs which were used in estimation until this point.
        """
        return self.cells_genes_pair_native_experssion_estimation[
            self.cells_genes_pair_native_experssion_estimation.max_relative_expression_for_pairs
            <= relative_expression_value
        ].index


class NoiseNativeExpressionEstimation(object):
    def __init__(
        self,
        batches_noise_estimation: pd.DataFrame,
        cells_genes_pair_native_experssion_estimation: pd.DataFrame,
        estimation_equations: pd.DataFrame,
        relative_expression_for_pairs: float,
    ) -> None:
        """Holds the final estimation for batch noise levels per umi depth with the shared native expressions values.

        Args:
            batches_noise_estimation (pd.DataFrame): The estimation of the batches noise levels.

            cells_genes_pair_native_experssion_estimation (pd.DataFrame): The estimation of the native expression.

            estimation_equations (pd.DataFrame): The equations which were used to estimate the noise levels and native expression.

            relative_expression_for_pairs (float): The relative epxression which were used for those equations.
            Meaning, what is the upper limit of relative expression which still yields pairs we used to build the equations.
        """

        self.cells_genes_pair_native_experssion_estimation = (
            cells_genes_pair_native_experssion_estimation
        )
        self.estimation_equations = estimation_equations

        self.relative_expression_for_pairs = relative_expression_for_pairs

        batches_labels = [
            i[0] for i in batches_noise_estimation.index.str.split("_umi_depth_bin_")
        ]
        umi_depth_bin_label = [
            i[1] for i in batches_noise_estimation.index.str.split("_umi_depth_bin_")
        ]
        batches_noise_estimation["umi_depth_bin"] = [
            int(i) for i in umi_depth_bin_label
        ]
        batches_noise_estimation.index = batches_labels

        self.batches_noise_estimation = batches_noise_estimation
