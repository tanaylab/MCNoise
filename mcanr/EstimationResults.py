"""
Hold the noise and native expression estimation, including functions helpful in plotting the data to help the user understand the performance of the estimation process.
This class is generated before a continuous estimation process is run, and after each step, a call for `add_estimation_step` is made to add more data to the class.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from adjustText import adjust_text
from mcanr import utilities


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
    ) -> float:
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

    def plot_number_of_batches_and_pgm_per_step(
        self, use_steps_as_axis: bool = True
    ) -> None:
        """
        Plot the number of batches and Pgm used in each step\ relative expression value.

        :param use_steps_as_axis: Should the x-axis be the number of steps, if false - present the relative expression value, defaults to True
        :type use_steps_as_axis: bool, optional
        """
        key = self.steps if use_steps_as_axis else self.steps_relative_expression

        combined_noise_level_estimation = pd.concat(
            self.noise_levels_estimations, keys=key
        )
        combined_noise_level_estimation.index.names = ["step", "batch"]
        combined_noise_level_estimation.reset_index(inplace=True)

        combined_native_expression_estimation = pd.concat(
            self.native_expression_estimations, keys=key
        )
        combined_native_expression_estimation.index.names = ["step", "Pgm"]
        combined_native_expression_estimation.reset_index(inplace=True)

        umi_depth_bins = self.noise_levels_estimations[-1].umi_depth_bin.unique()
        with sb.plotting_context(rc={"font.size": 30}):
            for umi_depth_bin in sorted(umi_depth_bins):
                plt.figure(figsize=(16, 8))
                combined_noise_level_estimation[
                    combined_noise_level_estimation.umi_depth_bin == umi_depth_bin
                ].groupby("step").nunique("batch")["batch"].plot(label="#Batches")
                combined_native_expression_estimation.groupby("step").nunique("Pgm")[
                    "Pgm"
                ].plot(label="#Pgm")

                plt.title("Number of batches and Pgm per step - bin %s" % umi_depth_bin)
                if not use_steps_as_axis:
                    plt.xlabel("relative expression value")

                plt.legend()
                plt.tight_layout()
                plt.show()

    def plot_noise_level_estimation_per_step(
        self, use_steps_as_axis: bool = True, show_batch_name: bool = False
    ) -> None:
        """
        Plot the estimation of different batches over the different steps\ relative expression.

        :param use_steps_as_axis: Should the x-axis be the number of steps, if false - present the relative expression value, defaults to True.
        :type use_steps_as_axis: bool, optional

        :param show_batch_name: Should we print the batch name as legend, defaults to False.
        :type show_batch_name: bool, optional
        """
        key = self.steps if use_steps_as_axis else self.steps_relative_expression
        xlabel = "steps" if use_steps_as_axis else "relative expression"
        ticks_values = (
            self.steps if use_steps_as_axis else self.steps_relative_expression
        )

        combined_noise_level_estimation = pd.concat(
            self.noise_levels_estimations, keys=key
        )
        combined_noise_level_estimation.index.names = ["step", "batch"]
        combined_noise_level_estimation.reset_index(inplace=True)

        umi_depth_bins = self.noise_levels_estimations[-1].umi_depth_bin.unique()
        with sb.plotting_context(rc={"font.size": 30}):
            for umi_depth_bin in umi_depth_bins:
                noise_levels_estimation_for_umi_depth = combined_noise_level_estimation[
                    combined_noise_level_estimation.umi_depth_bin == umi_depth_bin
                ]
                fig = plt.figure(figsize=(16, 8))
                ax = fig.add_subplot(
                    111,
                    xlabel=xlabel,
                    ylabel="Predicted noise (%)",
                    title="Predicted noise over steps - bin %s" % umi_depth_bin,
                    ylim=(
                        0,
                        combined_noise_level_estimation.predicted_percentages.max()
                        * 1.1,
                    ),
                    xlim=[ticks_values[0], ticks_values[-1]],  # type: ignore
                )

                for batch in sorted(
                    noise_levels_estimation_for_umi_depth.batch.unique()
                ):
                    ax.errorbar(
                        x=noise_levels_estimation_for_umi_depth[
                            noise_levels_estimation_for_umi_depth.batch == batch
                        ].step,
                        y=noise_levels_estimation_for_umi_depth[
                            noise_levels_estimation_for_umi_depth.batch == batch
                        ].predicted_percentages,
                        yerr=noise_levels_estimation_for_umi_depth[
                            noise_levels_estimation_for_umi_depth.batch == batch
                        ].predicted_sd_percentages,
                        capsize=5,
                        label=batch,
                        elinewidth=5,
                        lw=5,
                    )

                plt.grid(axis="y")
                if show_batch_name:
                    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", ncol=3)
                plt.show()

    def plot_native_expression_estimation_per_step(
        self, use_steps_as_axis: bool = True
    ) -> None:
        """
        Plot the estimation of different native expression over the different steps\ relative expression.

        :param use_steps_as_axis: Should the x-axis be the number of steps, if false - present the relative expression value, defaults to True.
        :type use_steps_as_axis: bool, optional
        """
        key = self.steps if use_steps_as_axis else self.steps_relative_expression
        xlabel = "steps" if use_steps_as_axis else "relative expression"
        ticks_values = (
            self.steps if use_steps_as_axis else self.steps_relative_expression
        )

        combined_native_expression_estimation = pd.concat(
            self.native_expression_estimations, keys=key
        )
        combined_native_expression_estimation.index.names = ["step", "pgm"]
        combined_native_expression_estimation.reset_index(inplace=True)

        with sb.plotting_context(rc={"font.size": 30}):
            fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(
                111,
                xlabel=xlabel,
                ylabel="Predicted native expression fraction",
                title="Predicted native expression over steps",
                ylim=(0, combined_native_expression_estimation.predicted.max() * 1.1),
                xlim=[ticks_values[0], ticks_values[-1]],  # type: ignore
            )

            for pgm in sorted(combined_native_expression_estimation.pgm.unique()):
                ax.errorbar(
                    x=combined_native_expression_estimation[
                        combined_native_expression_estimation.pgm == pgm
                    ].step,
                    y=combined_native_expression_estimation[
                        combined_native_expression_estimation.pgm == pgm
                    ].predicted,
                    yerr=combined_native_expression_estimation[
                        combined_native_expression_estimation.pgm == pgm
                    ].predicted_sd,
                    capsize=5,
                    label=pgm,
                    elinewidth=5,
                    lw=5,
                )

            plt.grid(axis="y")
            plt.show()

    def plot_noise_estimation_difference_between_umi_depth_bins(
        self,
        step: int,
        umi_depth_bin_1: int,
        umi_depth_bin_2: int,
    ) -> None:
        """
        Scatter plot of batch noise estimation between different umi depth bins.

        :param step: The required step for the noise estimation results
        :type step: int

        :param umi_depth_bin_1: The first umi depth bin to use as the x-axis
        :type umi_depth_bin_1: int

        :param umi_depth_bin_2: The second umi depth bin to use as the y-axis
        :type umi_depth_bin_2: int
        """
        assert step in self.steps, "There is no estimation for the requested step"
        noise_levels = self.noise_levels_estimations[self.steps.index(step)]

        umi_depth_bin = noise_levels.umi_depth_bin.unique()
        assert umi_depth_bin_1 in umi_depth_bin, (
            "No data for the umi depth bin %s in the requested step" % umi_depth_bin_1
        )
        assert umi_depth_bin_2 in umi_depth_bin, (
            "No data for the umi depth bin %s in the requested step" % umi_depth_bin_2
        )

        noise_levels_umi_depth_1 = noise_levels[
            noise_levels.umi_depth_bin == umi_depth_bin_1
        ]
        noise_levels_umi_depth_2 = noise_levels[
            noise_levels.umi_depth_bin == umi_depth_bin_2
        ]

        shared_batches = noise_levels_umi_depth_1.index & noise_levels_umi_depth_2.index

        noise_levels_umi_depth_1 = noise_levels_umi_depth_1.loc[shared_batches]
        noise_levels_umi_depth_2 = noise_levels_umi_depth_2.loc[shared_batches]

        texts = []

        with sb.plotting_context(rc={"font.size": 30}):
            plt.figure(figsize=(16, 8))

            plt.xlabel("Noise levels - bin %s" % umi_depth_bin_1)
            plt.ylabel(
                "Noise levels - bin %s" % umi_depth_bin_2,
            )
            plt.title("Predicted noise levels in differnt umi depth bins")

            plt.errorbar(
                x=noise_levels_umi_depth_1.predicted_percentages,
                y=noise_levels_umi_depth_2.predicted_percentages,
                xerr=noise_levels_umi_depth_1.predicted_sd_percentages,
                yerr=noise_levels_umi_depth_2.predicted_sd_percentages,
                fmt=".",
                ecolor="red",
                c="black",
                ms=8,
            )

            for batch in shared_batches:
                t = plt.text(
                    x=noise_levels_umi_depth_1.loc[batch].predicted_percentages,
                    y=noise_levels_umi_depth_2.loc[batch].predicted_percentages,
                    s=batch,
                )
                texts.append(t)

            adjust_text(
                texts,
                arrowprops=dict(arrowstyle="-", color="red", lw=0.5),
                expand_text=(1, 1),
            )

            min_value = min(
                0,
                min(
                    noise_levels_umi_depth_1.predicted_percentages.min(),
                    noise_levels_umi_depth_2.predicted_percentages.min(),
                ),
            )

            max_value = max(
                noise_levels_umi_depth_1.predicted_percentages.max(),
                noise_levels_umi_depth_2.predicted_percentages.max(),
            )

            max_sd = max(
                noise_levels_umi_depth_1.predicted_sd_percentages.max(),
                noise_levels_umi_depth_2.predicted_sd_percentages.max(),
            )
            plt.xlim(min_value, max_value + max_sd)
            plt.ylim(min_value, max_value + max_sd)

            plt.show()

    def plot_noise_estimation_delta_between_umi_depth_bins(
        self,
        step: int,
        umi_depth_bin_1: int,
        umi_depth_bin_2: int,
    ) -> None:
        """
        Scatter plot of the difference between batch noise estimation of different umi depth bins.

        :param step: The required step for the noise estimation results
        :type step: int

        :param umi_depth_bin_1: The first umi depth bin to use as the x-axis
        :type umi_depth_bin_1: int

        :param umi_depth_bin_2: The second umi depth bin to use as the y-axis
        :type umi_depth_bin_2: int
        """
        assert step in self.steps, "There is no estimation for the requested step"
        noise_levels = self.noise_levels_estimations[self.steps.index(step)]

        umi_depth_bin = noise_levels.umi_depth_bin.unique()
        assert umi_depth_bin_1 in umi_depth_bin, (
            "No data for the umi depth bin %s in the requested step" % umi_depth_bin_1
        )
        assert umi_depth_bin_2 in umi_depth_bin, (
            "No data for the umi depth bin %s in the requested step" % umi_depth_bin_2
        )

        noise_levels_umi_depth_1 = noise_levels[
            noise_levels.umi_depth_bin == umi_depth_bin_1
        ]
        noise_levels_umi_depth_2 = noise_levels[
            noise_levels.umi_depth_bin == umi_depth_bin_2
        ]

        shared_batches = noise_levels_umi_depth_1.index & noise_levels_umi_depth_2.index

        noise_levels_umi_depth_1 = noise_levels_umi_depth_1.loc[shared_batches]
        noise_levels_umi_depth_2 = noise_levels_umi_depth_2.loc[shared_batches]

        texts = []

        with sb.plotting_context(rc={"font.size": 30}):
            plt.figure(figsize=(16, 8))

            plt.xlabel("Noise levels - bin %s" % umi_depth_bin_1)
            plt.ylabel(
                "Delta between bin %s and bin %s" % (umi_depth_bin_2, umi_depth_bin_1)
            )
            plt.title("Predicted difference in noise levels between umi depth bins")

            plt.errorbar(
                x=noise_levels_umi_depth_1.predicted_percentages,
                y=noise_levels_umi_depth_2.predicted_percentages
                - noise_levels_umi_depth_1.predicted_percentages,
                xerr=noise_levels_umi_depth_1.predicted_sd_percentages,
                yerr=noise_levels_umi_depth_2.predicted_sd_percentages
                + noise_levels_umi_depth_1.predicted_sd_percentages,
                fmt=".",
                ecolor="red",
                c="black",
                ms=8,
            )

            for batch in shared_batches:
                t = plt.text(
                    x=noise_levels_umi_depth_1.loc[batch].predicted_percentages,
                    y=noise_levels_umi_depth_2.loc[batch].predicted_percentages
                    - noise_levels_umi_depth_1.loc[batch].predicted_percentages,
                    s=batch,
                )
                texts.append(t)

            adjust_text(
                texts,
                arrowprops=dict(arrowstyle="-", color="red", lw=0.5),
                expand_text=(1, 1),
            )

            max_delta = max(
                np.abs(
                    noise_levels_umi_depth_2.predicted_percentages
                    - noise_levels_umi_depth_1.predicted_percentages
                )
            ) + max(
                noise_levels_umi_depth_2.predicted_sd_percentages
                + noise_levels_umi_depth_1.predicted_sd_percentages
            )

            plt.ylim(-max_delta, max_delta)

            plt.show()

    
    def plot_noise_estimation_difference_between_results(self, other:object, step_self:int, step_other:int, x_label:str = "self noise estimation", y_label:str="other noise estimation") -> None:
        """
        Allow for comparison between different estimation results objects by plotting the different noise estimation of them one vs the other.

        :param other: The other estimation object we want to compare to ours
        :type other: NoiseNativeExpressionEstimation
        
        :param step_self: The estimation step for the current object 
        :type step_self: int

        :param step_other: The estimation step for the other objects
        :type step_other: int

        :param x_label: A label for the scatter plot x axis, defaults to "self_estimation"
        :type x_label: str, optional

        :param y_label: A label for the scatter plot y axis, defaults to "other_estimation"
        :type y_label: str, optional
        """
        assert step_self in self.steps, "There is no estimation for the requested step in the current estimation object"
        assert step_other in other.steps, "There is no estimation for the requested step in the other estimation object"

        self_noise_estimation = self.noise_levels_estimations[self.steps.index(step_self)]
        other_noise_estimation = other.noise_levels_estimations[other.steps.index(step_other)]

        for umi_depth_bin in sorted(self_noise_estimation.umi_depth_bin.unique()):
            self_noise_estimation_for_umi_depth_bin = self_noise_estimation[self_noise_estimation.umi_depth_bin == umi_depth_bin]
            other_noise_estimation_for_umi_depth_bin = other_noise_estimation[other_noise_estimation.umi_depth_bin == umi_depth_bin]
            shared_batches = self_noise_estimation_for_umi_depth_bin.index & other_noise_estimation_for_umi_depth_bin.index

            with sb.plotting_context(rc={"font.size": 30}):
                fig = plt.figure(figsize=(10, 10))
                _ = fig.add_subplot(
                    111,
                    xlabel=x_label,
                    ylabel=y_label,
                    title="Noise estimation bin %s" % umi_depth_bin,
                )

                plt.scatter(
                    x=self_noise_estimation_for_umi_depth_bin.loc[shared_batches].predicted_percentages,
                    y=other_noise_estimation_for_umi_depth_bin.loc[shared_batches].predicted_percentages,
                    s=100,
                )
                plt.grid()
                plt.tight_layout()
                plt.show()
        
    def plot_native_expression_difference_between_results(self, other:object, step_self:int, step_other:int, x_label:str = "self native expression", y_label:str="other native expression") -> None:
        """
        Allow for comparison between different estimation results objects by plotting the different native expression of them one vs the other.

        :param other: The other estimation object we want to compare to ours
        :type other: NoiseNativeExpressionEstimation
        
        :param step_self: The estimation step for the current object 
        :type step_self: int

        :param step_other: The estimation step for the other objects
        :type step_other: int

        :param x_label: A label for the scatter plot x axis, defaults to "self_estimation"
        :type x_label: str, optional

        :param y_label: A label for the scatter plot y axis, defaults to "other_estimation"
        :type y_label: str, optional
        """
        assert step_self in self.steps, "There is no estimation for the requested step in the current estimation object"
        assert step_other in other.steps, "There is no estimation for the requested step in the other estimation object"

        self_native_expression = self.native_expression_estimations[self.steps.index(step_self)]
        other_native_expression = other.native_expression_estimations[other.steps.index(step_other)]

        shared_pgm = self_native_expression.index & other_native_expression.index

        with sb.plotting_context(rc={"font.size": 30}):
            fig = plt.figure(figsize=(10, 10))
            _ = fig.add_subplot(
                111,
                xlabel=x_label,
                ylabel=y_label,
                title="Native expression comparison",
            )

            plt.scatter(
                x=self_native_expression.loc[shared_pgm].predicted,
                y=other_native_expression.loc[shared_pgm].predicted,
                s=100,
            )
            
            plt.grid()
            plt.tight_layout()
            plt.yscale("symlog", linthreshy=1e-5, basey=10)
            plt.xscale("symlog", linthreshx=1e-5, basex=10)
            plt.xlim(0)
            plt.ylim(0)
            plt.show()

    def plot_observed_vs_predicted_umis(self, step: int) -> None:
        """
        Produce a scatter plot of the predicted noisy umis vs the observed noisy umis.

        :param step: The prediction from this step will be used as the native expression and noise levels estimation.
        :type step: int
        """
        assert step in self.steps, "There is no estimation for the requested step"

        equations = pd.concat(self.estimation_equations[self.steps.index(step)]).fillna(
            0
        )
        noise_levels = self.noise_levels_estimations[self.steps.index(step)].copy()
        noise_levels.index = (
            noise_levels.index
            + "_umi_depth_bin_"
            + noise_levels.umi_depth_bin.astype(str)
        )
        native_expression_estimation = self.native_expression_estimations[
            self.steps.index(step)
        ]
        coefficents = pd.concat(
            [noise_levels.predicted, native_expression_estimation.predicted]
        )
        coefficents = coefficents.loc[equations.columns[1:]]

        observed_values = equations.observed

        predicted = np.sum(
            equations[equations.columns[1:]] * coefficents.values, axis=1
        )

        with sb.plotting_context(rc={"font.size": 30}):
            fig = plt.figure(figsize=(10, 10))
            fig.add_subplot(
                111,
                xlabel="Predicted umi count",
                ylabel="Observed umi count",
                title="Observed vs Predicted",
                ylim=(1, max(np.max(observed_values), np.max(predicted)) * 1.1),
                xlim=(1, max(np.max(observed_values), np.max(predicted)) * 1.1),
            )

            plt.scatter(x=predicted, y=observed_values, s=5)
            plt.plot(
                [1, max(np.max(observed_values), np.max(predicted)) * 1.1],
                [1, max(np.max(observed_values), np.max(predicted)) * 1.1],
                "k:",
            )

            plt.yscale("log", basey=2)
            plt.xscale("log", basex=2)
            plt.grid()
            plt.show()

    def plot_log_likelihood_of_estimation(
        self,
        step: int,
        min_noise_estimation: float = 0,
        max_noise_estimation: float = 0.2,
        noise_estimation_step_size=0.01,
        n_cols: int = 6,
    ) -> None:
        """
        Show the log-likelihood value of our current estimation, and compare it to other possible ones, using the same native expression which was already found.
        This is used to check how tight our estimation and the degree of local\global min this estimation is.
        The set of estimation we are going to go through is from `min_noise_estimation` to `max_noise_estimation` in jumps of `noise_estimation_step_size`

        :param step: _description_
        :type step: int

        :param min_noise_estimation: The smallest noise estimation to check, defaults to 0.
        :type min_noise_estimation: float, optional

        :param max_noise_estimation: The highest noise estimation to check, defaults to 0.2.
        :type max_noise_estimation: float, optional

        :param noise_estimation_step_size: The delta between different noise level estimation values to check, defaults to 0.01.
        :type noise_estimation_step_size: float, optional

        :param n_cols: Number of columns for the log likelihood plot, defaults to 6.
        :type n_cols: int, optional
        """
        assert step in self.steps, "There is no estimation for the requested step"

        equations = pd.concat(self.estimation_equations[self.steps.index(step)]).fillna(
            0
        )
        noise_levels = self.noise_levels_estimations[self.steps.index(step)].copy()
        noise_levels.index = (
            noise_levels.index
            + "_umi_depth_bin_"
            + noise_levels.umi_depth_bin.astype(str)
        )
        native_expression_estimation = self.native_expression_estimations[
            self.steps.index(step)
        ]
        coefficents = pd.concat(
            [noise_levels.predicted, native_expression_estimation.predicted]
        ).copy()
        coefficents = coefficents.loc[equations.columns[1:]]
        batches_locations = ~coefficents.index.str.startswith("Pgm")

        observed_values = equations.observed

        likelihood_df = pd.DataFrame(
            index=coefficents[batches_locations].index,
            columns=np.arange(
                min_noise_estimation, max_noise_estimation, noise_estimation_step_size
            ),
        )

        for i, current_noise in enumerate(
            np.arange(
                min_noise_estimation, max_noise_estimation, noise_estimation_step_size
            )
        ):
            coefficents[batches_locations] = current_noise
            predicted = np.sum(
                equations[equations.columns[1:]] * coefficents.values, axis=1
            )
            likelihood = utilities.calculate_negative_loglikelihood(
                observed=observed_values, predicted=predicted
            )
            likelihood = likelihood.replace(
                -np.inf, np.min(likelihood[likelihood > -np.inf])
            )
            for j, batch in enumerate(coefficents[batches_locations].index):
                batch_equations = equations.loc[:, batch] != 0
                likelihood_df.iloc[j, i] = np.sum(likelihood[batch_equations])

        likelihood_df *= -1

        with sb.plotting_context(rc={"font.size": 30}):
            num_rows = likelihood_df.shape[0] / n_cols
            num_rows = int(num_rows) if int(num_rows) == num_rows else int(num_rows) + 1
            _, axes = plt.subplots(
                num_rows,
                n_cols,
                figsize=(12 * n_cols, 10 * num_rows),
            )

            axes = axes.reshape(-1)
            plt.tight_layout()
            for i, batch_bin in enumerate(sorted(likelihood_df.index)):
                ax = axes[i]
                ax.scatter(
                    x=likelihood_df.columns * 100,
                    y=likelihood_df.loc[batch_bin].values,
                    s=200,
                )

                bin_results = noise_levels.loc[batch_bin]

                ax.axvline(bin_results.predicted * 100, c="b", linestyle="--", lw=4)
                ax.axvline(
                    (bin_results.predicted + bin_results.predicted_sd) * 100,
                    c="r",
                    linestyle=":",
                    lw=4,
                )
                ax.axvline(
                    (bin_results.predicted - bin_results.predicted_sd) * 100,
                    c="r",
                    linestyle=":",
                    lw=4,
                )

                start = int(np.log2(max(likelihood_df.loc[batch_bin].values.min(), 1)))
                ax.set_yscale("log", basey=2)
                ax.set_ylim(2**start, 2 ** (start + 8))
                ax.set_yticks([2 ** (start + i) for i in range(0, 8, 2)])
                ax.set_title("%s" % batch_bin)

            plt.subplots_adjust(
                left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.5, hspace=0.3
            )

            plt.suptitle("- Log-Likelihood on different noise estimation", fontsize=100)
            plt.show()
