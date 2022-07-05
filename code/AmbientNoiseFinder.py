"""
AmbientNoiseFinder

Object that wrap all the needed data structures and functions needed to identify ambient noise existance in a dataset.
This include the cells and metacells information, the empty droplets information per batch.
Using those data structures this module able to identify and point to combinations of genes inside metacells which are more likely to be originated from noise, or mostly originated from noise.
"""

from typing import Callable

import anndata as ad
import metacells as mc
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import distance
from sklearn.cluster import KMeans

from EstimationResults import NoiseNativeExpressionEstimation


class AmbientNoiseFinder(object):
    def __init__(
        self,
        cells_adata: ad.AnnData,
        metacells_adata: ad.AnnData,
        batches_empty_droplets_dict: dict[str : pd.Series],
        extract_batch_names_function: Callable[[ad.AnnData], list[str]],
        umi_depth_number_of_bins=3,
        umi_depth_min_percentile=5,
        umi_depth_max_percentile=95,
        expression_delta_for_significant_genes=4,
        number_of_genes_clusters=10,
        minimum_genes_in_cluster=1,
        number_of_metacells_clusters=10,
        minimum_metacells_in_cluster=5,
        genes_clusters=None,
        metacells_clusters=None,
    ) -> None:
        """Holds all the data needed to find ambient noise traces.
        This data includes the cell and metacells information, the empty droplets information for the different batches.
        Using this data, this object allow to identify noise-prone pairs of metacells and genes clusters which will then be used to estimate the noise levels.

        The init function flow:
        1. Make sure that the genes in the cells, metacells and empty droplets information are the same - in many cases this isn't true.
        This mismatch might happened by using different versions of cell ranger which yield different genes names, but also by having genes id missing in one data file while still existing on the other.
        In general this shouldn't happened but probably happened due to using different versions of files or pipeline proccesses.

        2. Extract basic information from the cells and metacells objects and add it as properties of the addata object.
        For example: the umi depth of each cells, the batch of the cell, the umi depth bin of it.

        3. Cluster the metacells and the most significant genes.
        Here the user can either provide number of clusters to generate or provide the clustering data istself, for example: gene modules and metacells annotation.

        4. For each gene cluster, rank the metacells clusters by the likelihood of having the majority of observed umis in this metacell-genes clusters from noise and not native expression.

        Args:
            cells_adata (ad.AnnData): The full annotated data of the cells.

            metacells_adata (ad.AnnData): The full annotated data of the metacells.

            batches_empty_droplets_dict (dict[str: pd.Series]): Mapping between the batch name and the empty droplets distribution across the genes.

            extract_batch_names_function (Callable[[ad.AnnData], list[str]]): Function that take the cells addata file and return the batch for each cell.

            umi_depth_number_of_bins (int, optional): Number of bins to split the cells. This seperation will produce equall number of cells in each bin based on the umi depth.
            Defaults to 3.

            umi_depth_min_percentile (int, optional): Bottom percentile for cells size to remove as outliers. Defaults to 5.

            umi_depth_max_percentile (int, optional): Top percentile for cells size to remove as outliers. Defaults to 95.

            expression_delta_for_significant_genes (int, optional): The minimal expression difference between metacells to be considered a significant gene. Defaults to 4.

            number_of_genes_clusters (int, optional): How many genes clusters should we produce. Defaults to 10.

            minimum_genes_in_cluster (int, optional): Mark genes clusters with too few genes inside which we might want to ignore in the future. Defaults to 1.

            number_of_metacells_clusters (int, optional): How many metacells clusters should we produce. Defaults to 10.

            minimum_metacells_in_cluster (int, optional):  Mark metacells clusters with too few metacells inside which we might want to ignore in the future. Defaults to 5.

            genes_clusters (pd.Series, optional): A series with each gene and the cluster it should be. This allows the user to provide genes modules instead of auto clusters.
            Defaults to None.

            metacells_clusters (pd.Series, optional): A serues with each metacell id and the cluster it should be. This allows the user to provide annotaiton information instead of auto clustering.
             Defaults to None.

        """

        self.umi_depth_number_of_bins = umi_depth_number_of_bins
        self.batches = list(batches_empty_droplets_dict.keys())

        self.cells_adata = cells_adata[~cells_adata.obs.outlier]

        self.metacells_adata = metacells_adata

        self.batches_empty_droplets_dict = self._filter_out_missing_genes(
            batches_empty_droplets_dict
        )

        self.cells_adata.obs["umi_depth"] = pd.Series(
            data=mc.ut.get_o_numpy(self.cells_adata, name="__x__", sum=True),
            index=cells_adata.obs.index,
        )
        self.cells_df = mc.ut.get_vo_frame(self.cells_adata)

        self.umi_depth_bins_thresholds = self._get_umi_depth_bin_threshold_list(
            umi_depth_number_of_bins=umi_depth_number_of_bins,
            max_percentile=umi_depth_max_percentile,
            min_percentile=umi_depth_min_percentile,
        )

        self._add_umi_depth_bins_information_to_cells_adata()
        self._add_effective_umi_depth_for_cells()
        self._add_batches_names_to_cells_adata(extract_batch_names_function)

        self.metacells_df = mc.ut.get_vo_frame(self.metacells_adata)

        self.metacells_log_fractions = np.log2(
            self.metacells_df.divide(self.metacells_df.sum(axis=1), axis=0) + 1e-5
        )

        if genes_clusters == None:
            genes_clusters, self.small_genes_clusters = self._get_genes_clusters(
                expression_delta_for_significant_genes=expression_delta_for_significant_genes,
                number_of_clusters=number_of_genes_clusters,
                minimum_genes_in_cluster=minimum_genes_in_cluster,
            )

        self._add_geness_clusters_information(genes_clusters)

        if metacells_clusters == None:
            (
                metacells_clusters,
                self.small_metacells_clusters,
            ) = self._get_metacells_clusters(
                number_of_clusters=number_of_metacells_clusters,
                minimum_metacells_in_cluster=minimum_metacells_in_cluster,
            )

        self.metacells_adata.obs["metacells_cluster"] = metacells_clusters.values
        self._add_cells_clusters_information()

        self.metacells_genes_pair_relative_expression_df = (
            self._calculate_metacells_genes_pair_relative_expression_to_max()
        )
        self.empty_droplet_genes_cluster_fraction = (
            self._get_empty_droplet_genes_cluster_fraction()
        )

    def get_cells_adata_with_noise_level_estimations(
        self, estimations_results: NoiseNativeExpressionEstimation
    ) -> ad.AnnData:
        """
        Add noise levels estimation to the cells adata - this includes the noise levels and the umi depth bin for this cells.
        If the cells are above or below a umi depth bins, or no batch estimation was provided for this umi depth bin we will use the closes estimation for this bin.

        Args:
            estimations_results (NoiseNativeExpressionEstimation): Holds the full results of the estimation after the entire ambient noise estimatoin pipeline.
            This object should have been generated by the same instance of the AmbientNoiseFinder object that is being called.

        Returns:
            ad.AnnData: The full annotated data of the cells, now with noise estimation information.
        """
        # Add umi depth bin information for cells with too much or too little umis. Using the closest umi depth bin for them.
        self.cells_adata.obs.loc[
            self.cells_adata.obs.umi_depth <= self.umi_depth_bins_thresholds[0],
            "umi_depth_bin",
        ] = 1

        self.cells_adata.obs.loc[
            self.cells_adata.obs.umi_depth >= self.umi_depth_bins_thresholds[-1],
            "umi_depth_bin",
        ] = self.umi_depth_number_of_bins

        self.cells_adata.obs.loc[:, "batch_estimated_noise"] = 0

        for batch_name in self.cells_adata.obs.batch.unique():
            if (
                batch_name
                not in estimations_results.batches_noise_estimation.index.unique()
            ):
                continue

            for umi_depth_bin in range(1, self.umi_depth_number_of_bins):
                closest_umi_depth_bin = umi_depth_bin
                relavent_batch_estimation = (
                    estimations_results.batches_noise_estimation.loc[batch_name]
                )

                # If only one estimation for this batch we can only take it as this is the closest one.
                if len(relavent_batch_estimation.shape) == 1:
                    noise_level_estimation = relavent_batch_estimation.predicted

                else:
                    # If more then one options, find the closest one based on umi depth, prefer to take bigger umi depth then smaller one.
                    available_umis_delta = np.abs(
                        closest_umi_depth_bin
                        - relavent_batch_estimation.umi_depth_bin.unique()
                    )
                    closest_umi_depth_bin = np.max(
                        relavent_batch_estimation.umi_depth_bin.unique()[
                            available_umis_delta == np.min(available_umis_delta)
                        ]
                    )
                    noise_level_estimation = relavent_batch_estimation[
                        relavent_batch_estimation.umi_depth_bin == closest_umi_depth_bin
                    ].predicted[0]

                self.cells_adata.obs.loc[
                    (self.cells_adata.obs.batch == batch_name)
                    & (self.cells_adata.obs.umi_depth_bin == umi_depth_bin),
                    "noise_level_estimation",
                ] = noise_level_estimation

        return self.cells_adata

    def _add_effective_umi_depth_for_cells(self):
        # TODO: check this with Oren
        """
        Adding a column for each cell with the effective umi depth in the metacell.
        Cells with umi depth higher then twice the median are being linearly adjusted and we would like to use the corrected umi depth when calculating the noise added by those cells.
        """
        metacells_median_umi_depth = self.cells_adata.obs.groupby("metacell").agg(
            {"umi_depth": np.median}
        )
        self.cells_adata.obs["effective_umi_depth"] = np.min(
            [
                self.cells_adata.obs.umi_depth.values.reshape(-1, 1),
                metacells_median_umi_depth.loc[self.cells_adata.obs.metacell] * 2,
            ],
            axis=0,
        )

    def _add_batches_names_to_cells_adata(
        self, extract_batch_names_function: Callable[[ad.AnnData], list[str]]
    ):
        """Extract batch names for each cell and add this to the cells adata.

        Args:
           extract_batch_names_function (Callable[[ad.AnnData], list[str]]): Function that take the cells addata file and return the batch for each cell.
        """
        self.cells_adata.obs["batch"] = extract_batch_names_function(self.cells_adata)

    def _filter_out_missing_genes(
        self, batches_empty_droplets_dict: dict[str : pd.Series]
    ) -> dict[str : pd.Series]:
        """Make sure that the cells anndata and the metacell anndata share the same genes with the empty droplets files.
        In many cases this might not be true and then we clip those objects without them.
        Most likely this is happening due to different versions of pipleines run over those different files.

        Args:
            batches_empty_droplets_dict - Mapping between the batch name and the empty droplets distribution series.

        Returns:
            dict[str:pd.Series]: Mapping between the batch name and the empty droplets distribution series, now with the same genes as cells and metacells.
        """
        common_genes = self.cells_adata.var.index & self.metacells_adata.var.index
        for batch in batches_empty_droplets_dict:
            common_genes = common_genes & batches_empty_droplets_dict[batch].index

        self.cells_adata = self.cells_adata[:, common_genes]
        self.metacells_adata = self.metacells_adata[:, common_genes]
        batches_empty_droplets_dict = {
            batch: batches_empty_droplets_dict[batch].loc[common_genes]
            for batch in batches_empty_droplets_dict
        }
        return batches_empty_droplets_dict

    def _get_significant_genes(
        self, expression_delta_for_significant_genes: float
    ) -> pd.Index:
        """Extract all genes with enough expression difference between different metacells.
        We prefer not to use only one metacells to calculate this delta but to use percentile to make sure we don't estimate the noise based on small number of metacells or unique phenomena in the data.

        Args:
            expression_delta_for_significant_genes (float): The minimal expression difference between metacells to be considered a significant gene.

        Returns:
            pd.Index: A set of genes names, each considered significant and might allow for identification of ambient noise.
        """
        significant_genes = self.metacells_log_fractions.columns[
            np.percentile(self.metacells_log_fractions, 95, axis=0)
            - np.percentile(self.metacells_log_fractions, 5, axis=0)
            >= expression_delta_for_significant_genes
        ]
        return significant_genes

    def _get_genes_clusters(
        self,
        expression_delta_for_significant_genes: float,
        number_of_clusters: int,
        minimum_genes_in_cluster: int,
    ) -> pd.Series:
        """Perform hierarchical clustering of the significant genes to generate genes module like objects.
        We mark genes clusters which are below the required number as small_genes which the user might be able to use or not use.
        We don't combine genes moudles together to make sure those clusters have enough genes to actually allow the user to use well defined small genes clusters which
        will be strong enough and to prevent situation which we combine two genes modules and harm our ability to identify genes-metacells clusters.

        Args:
            expression_delta_for_significant_genes (float): The minimal expression difference between metacells to be considered a significant gene.

            number_of_clusters (int): How many genes clusters should we produce.

            minimum_genes_in_cluster (int): Genes clusters with less then this number of genes will be considered small.

        Returns:
            pd.Series: Each row is the name of the gene and the value is the gene cluster matching it.
        """
        significant_genes = self._get_significant_genes(
            expression_delta_for_significant_genes
        )

        genes_linkeage = hierarchy.linkage(
            distance.pdist(self.metacells_log_fractions.loc[:, significant_genes].T),
            method="average",
        )

        flat_cluster = fcluster(
            genes_linkeage, t=number_of_clusters, criterion="maxclust"
        )

        clusters_df = pd.DataFrame(
            {"cluster": flat_cluster, "gene": significant_genes},
            index=significant_genes,
        )
        cluster_series = clusters_df.set_index(["gene"]).squeeze()

        clusters_size = clusters_df.groupby("cluster").count()
        small_genes_clusters = clusters_size.index[
            np.where(clusters_size < minimum_genes_in_cluster)[0]
        ]

        return cluster_series, small_genes_clusters

    def _get_metacells_clusters(
        self, number_of_clusters: int, minimum_metacells_in_cluster: int
    ) -> pd.Series:
        """Perform kmeans clustering over the metacells.

        Args:
            number_of_clusters (int): How many metacells clusters should we produce.
            minimum_metacells_in_cluster (int): Clusters with fewer metacells will be marked as small_metacells clusters.

        Returns:
            pd.Series: Each row is the id of the metacells and the value is the cluster matching it.
        """
        kmeans = KMeans(n_clusters=number_of_clusters, random_state=0, n_init=10).fit(
            self.metacells_log_fractions[
                self.metacells_adata.var[
                    self.metacells_adata.var.genes_cluster != -1
                ].index
            ]
        )
        clusters_df = pd.DataFrame(
            {
                "metacell": self.metacells_df.index.astype(np.int),
                "cluster": kmeans.labels_.astype(np.int),
            },
            index=self.metacells_df.index.astype(np.int),
        )
        clusters_size = clusters_df.groupby("cluster").count()
        cluster_series = clusters_df.set_index(["metacell"]).squeeze()

        small_metacells_clusters = clusters_size.index[
            np.where(clusters_size < minimum_metacells_in_cluster)[0]
        ]

        return cluster_series, small_metacells_clusters

    def _add_cells_clusters_information(self):
        """For each cell in the adata object add the cluster id matching the metacell it belongs to."""
        self.cells_adata.obs["cells_cluster"] = -1
        for i in self.metacells_adata.obs.index:
            self.cells_adata.obs.loc[
                self.cells_adata.obs.metacell == int(i), "cells_cluster"
            ] = self.metacells_adata.obs.loc[i, "metacells_cluster"]

    def _add_geness_clusters_information(self, genes_clusters: pd.Series):
        """For each gene in the cells and metacells adata add the cluster id to it (or -1 if not in significant gene)

        Args:
            genes_clusters (pd.Series): Each row is the gene name and the value is the cluster id.
        """
        self.metacells_adata.var["genes_cluster"] = -1
        self.metacells_adata.var.loc[
            genes_clusters.index, "genes_cluster"
        ] = genes_clusters.values
        self.cells_adata.var["genes_cluster"] = -1
        self.cells_adata.var.loc[
            genes_clusters.index, "genes_cluster"
        ] = genes_clusters.values

    def _calculate_metacells_genes_pair_relative_expression_to_max(
        self,
    ) -> pd.DataFrame:
        """We want to find pair of metacells-genes clusters which are more likely to be noise oriented. This mean that the majority of the observed UMIs origineted from the ambient noise.
        To do so, we look at the expression for each pair and compare it to the maximum expression of all other metacells clusters across the same genes cluster, calculating the log fold.
        Metacells clusters which are LOW (negative fold) compare to some other metacells clusters for the same genes cluster can be defined as those noise prone pairs and they are candidates for noise level estimation.

        Returns:
            pd.DataFrame: A 2d dataframe with rows as metacells clusters, columns as genes clusers and the value in each place is the log fold of this location vs the maximum expression in the same column.
        """

        genes_clusters = self.cells_adata.var.genes_cluster.unique()[
            self.cells_adata.var.genes_cluster.unique() != -1
        ]
        metacells_clusters = self.cells_adata.obs.cells_cluster.unique()[
            self.cells_adata.obs.cells_cluster.unique() != -1
        ]

        # Direct DataFrame groupby(data)
        expressions_df = pd.DataFrame(
            columns=sorted(genes_clusters),
            index=sorted(metacells_clusters),
            dtype=np.float64,
        )

        for gene_cluster in genes_clusters:
            for metacell_cluster in metacells_clusters:
                expressions_df.loc[metacell_cluster, gene_cluster] = np.median(
                    np.log2(
                        self.metacells_df.loc[
                            self.metacells_adata.obs[
                                self.metacells_adata.obs.metacells_cluster
                                == metacell_cluster
                            ].index,
                            self.metacells_adata.var[
                                self.metacells_adata.var.genes_cluster == gene_cluster
                            ].index,
                        ]
                        + 1e-5
                    )
                )

        return expressions_df - expressions_df.max(axis=0)

    def _get_umi_depth_bin_threshold_list(
        self,
        umi_depth_number_of_bins=3,
        min_percentile=5,
        max_percentile=95,
    ) -> list[int]:
        """Generate a list representing the different umi depth bins intervals based on the given cells total umi counts in this dataset.
        The user can define how many umi depth bins we need and the bottom/top percentile of cells to remove to make sure we don't use outliers cells.

        Args:
            umi_depth_number_of_bins (int, optional): Number of umi depth bins to split the cells into. Defaults to 3.

            min_percentile (int, optional): Bottom percentile for cells size to remove as outliers. Defaults to 5.

            max_percentile (int, optional): Top percentile for cells size to remove as outliers. Defaults to 95.

        Returns:
            list[int]: A umi depth threshold intervals representation: [smallest_size, 1st bin end, second bin end, ..., largest size]
        """
        # Filter out top and low percentile to remove outliers
        total_umis_min, total_umis_max = np.percentile(
            self.cells_adata.obs["umi_depth"], [min_percentile, max_percentile]
        )

        valid_sizes = self.cells_adata.obs["umi_depth"][
            (self.cells_adata.obs["umi_depth"] >= total_umis_min)
            & (self.cells_adata.obs["umi_depth"] <= total_umis_max)
        ]

        # Split the remaining data to the different bins such that the number of cells in each bin is the mostly the same.
        bins_threshold_list = (
            np.arange(umi_depth_number_of_bins + 1) / umi_depth_number_of_bins
        )
        bins_threshold_list = np.quantile(valid_sizes, bins_threshold_list)

        return bins_threshold_list

    def _add_umi_depth_bins_information_to_cells_adata(self):
        """Go over the cells adata object and add the umi depth bin id for each cell row."""
        for i in range(len(self.umi_depth_bins_thresholds) - 1):
            min_umi_depth, max_umi_depth = (
                self.umi_depth_bins_thresholds[i],
                self.umi_depth_bins_thresholds[i + 1],
            )
            self.cells_adata.obs.loc[
                (self.cells_adata.obs.umi_depth > min_umi_depth)
                & (self.cells_adata.obs.umi_depth <= max_umi_depth),
                "umi_depth_bin",
            ] = (
                i + 1
            )

    def _get_empty_droplet_genes_cluster_fraction(self) -> pd.DataFrame:
        """Get a 2d representing of the fraction of empty droplets per gene cluster in a batch.
        Genes clusters are independent of the metacells clusters, but they depend on the different batch we use -> this is a different ambient noise distribution based on batches.

        Returns:
            pd.DataFrame: A 2d matrix with rows as batch, columns as genes clusers and the value in each place is the fractions of this gene cluster expression out of the whole umis in the ambient noise.
        """
        genes_clusters = self.metacells_adata.var.genes_cluster.unique()
        genes_clusters = genes_clusters[genes_clusters != -1]

        empty_droplet_genes_cluster_fraction = pd.DataFrame(
            index=self.batches_empty_droplets_dict.keys(),
            columns=genes_clusters,
            dtype=np.float64,
        )
        for genes_cluster in genes_clusters:
            genes = self.metacells_adata.var.genes_cluster[
                self.metacells_adata.var.genes_cluster == genes_cluster
            ].index
            for batch in self.batches_empty_droplets_dict:
                batch_empty_droplet_fraction = (
                    self.batches_empty_droplets_dict[batch].loc[genes].sum()
                    / self.batches_empty_droplets_dict[batch].sum()
                )

                empty_droplet_genes_cluster_fraction.loc[
                    batch, genes_cluster
                ] = batch_empty_droplet_fraction[0]

        return empty_droplet_genes_cluster_fraction
