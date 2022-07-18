"""
Hold the noise and native expression estimation, including functions helpful in plotting the data to help the user understand the performance of the estimation process.
This class is generated before a continuous estimation process is run, and after each step, a call for `add_estimation_step` is made to add more data to the class.
"""

import pandas as pd
import numpy as np


class NoiseNativeExpressionEstimation(object):
    def __init__(
        self,
        max_relative_expression_for_metacells_genes_to_use: float,
        number_of_steps: int,
        min_expected_umi_threshold: int,
        min_number_of_pgm_clusters_per_batch: int,
        min_number_of_batches_per_pgm_cluster: int,
    ) -> None:
        """
        Hold the noise and native expression estimation, including functions helpful in plotting the data to help the user understand the performance of the estimation process.
        This class is generated before a continuous estimation process is run, and after each step, a call for `add_estimation_step` is made to add more data to the class.

        :param max_relative_expression_for_metacells_genes_to_use: Store the max_relative_expression_for_metacells_genes_to_use which was defined by the user.
        :type max_relative_expression_for_metacells_genes_to_use: float

        :param number_of_steps: Store the number of steps the user requested.
        :type number_of_steps: int

        :param min_expected_umi_threshold: Store the minimal number of expected umis to make and equation valid.
        :type min_expected_umi_threshold: int

        :param min_number_of_pairs_per_batch: Store the minimum number of pairs per batch the user requested.
        :type min_number_of_pairs_per_batch: int

        :param min_number_of_batches_per_pair: Store the minimum number of batches per pair the user requested.
        :type min_number_of_batches_per_pair: int
        """
        self.max_relative_expression_for_metacells_genes_to_use = (
            max_relative_expression_for_metacells_genes_to_use
        )
        self.number_of_steps = number_of_steps
        self.min_expected_umi_threshold = min_expected_umi_threshold
        self.min_number_of_pairs_per_batch = min_number_of_pgm_clusters_per_batch
        self.min_number_of_batches_per_pair = min_number_of_batches_per_pgm_cluster

        self.steps: list[int] = []
        self.steps_relative_expression: list[float] = []
        self.estimation_equations: list[pd.DataFrame] = []
        self.noise_levels_estimations: list[pd.DataFrame] = []
        self.native_expression_estimations: list[pd.DataFrame] = []

    def add_estimation_step(
        self,
        noise_native_expression_estimation: pd.DataFrame,
        equations: pd.DataFrame,
        step: int,
        step_relative_expression: float,
    ):
        """
        Add and store the information of a specific step of estimation. This contains the number of steps and the current relative expression used.
        Also contains the equations which were used to estimate the noise and native expression and the estimation itself.

        :param noise_native_expression_estimation: The estimations results, this is a dataframe with the predicted value and std of the noise and native expression.
        :type noise_native_expression_estimation: pd.DataFrame

        :param equations: The equations which were used to estimate the noise and native expression.
        :type equations: pd.DataFrame

        :param step: The current step of this estimation.
        :type step: int

        :param step_relative_expression: The relative expression corresponding to the current step.
        :type step_relative_expression: float
        """
        self.steps.append(step)
        self.steps_relative_expression.append(step_relative_expression)
        self.estimation_equations.append(equations)

        # Split the results to two dataframes, one for batches and another for native expression.
        batches_noise_estimation_df = noise_native_expression_estimation[
            ~noise_native_expression_estimation.index.str.startswith("Pgm")
        ]
        batches_noise_estimation_df["predicted_percentages"] = (
            batches_noise_estimation_df["predicted"] * 100
        )
        batches_noise_estimation_df["predicted_sd_percentages"] = (
            batches_noise_estimation_df["predicted_sd"] * 100
        )

        # Change the index to the batch name and add another column for the umi depth bin.
        batches_labels = [
            i[0] for i in batches_noise_estimation_df.index.str.split("_umi_depth_bin_")
        ]
        umi_depth_bins = [
            int(i[1])
            for i in batches_noise_estimation_df.index.str.split("_umi_depth_bin_")
        ]

        batches_noise_estimation_df["umi_depth_bin"] = umi_depth_bins
        batches_noise_estimation_df.index = batches_labels

        self.noise_levels_estimations.append(batches_noise_estimation_df)

        # Split the dataframe of the native expression.
        cells_genes_pair_native_expression_estimation_df = (
            noise_native_expression_estimation[
                noise_native_expression_estimation.index.str.startswith("Pgm")
            ]
        )

        self.native_expression_estimations.append(
            cells_genes_pair_native_expression_estimation_df
        )

    def get_noise_estimation_for_batch_by_step_and_umi_depth_bin(
        self, batch: str, step: int, umi_depth_bin: int
    ) ->float:
        """
        Get the best noise estimation for a specific batch, step and umi depth bin.
        If there is no estimation for this batch - return 0.
        If there is several estimation for this batch but none for the requested umi depth bin, will take the closest one.

        :param batch: The name of the batch to get the estimation of.
        :type batch: str

        :param step: The requested estimation step.
        :type step: int

        :param umi_depth_bin: The umi depth bin of the requested batch.
        :type umi_depth_bin: int

        :return: The estimation, or closest estimation for this batch matching the requested variables or 0 if none exists.
        :rtype: float
        """
        assert step in self.steps, "There is no estimation for the requested step"
        
        step_index = self.steps.index(step)
        step_noise_levels_estimations = self.noise_levels_estimations[step_index]

        if batch not in step_noise_levels_estimations.index:
            return 0

        batch_estimation = step_noise_levels_estimations.loc[batch]

        # Only one value for this batch, take it
        if len(batch_estimation.shape) == 1:
            return batch_estimation.predicted

        # Several results, take closest one to the requested umi_depth_bin
        max_closest_umi_depth = np.max(
            np.where(
                np.abs(batch_estimation.umi_depth_bin - umi_depth_bin)
                == np.abs(batch_estimation.umi_depth_bin - umi_depth_bin).min()
            )
        )
        return batch_estimation.iloc[max_closest_umi_depth].predicted
