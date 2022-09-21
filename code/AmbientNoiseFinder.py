"""
AmbientNoiseFinder

Object that wrap all the needed data structures and functions needed to identify ambient noise existance in a dataset.
This include the cells and metacells information, the empty droplets information per batch.
Using those data structures this module able to identify and point to combinations of genes inside metacells which are more likely to be originated from noise, or mostly originated from noise.
"""

from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sb

import anndata as ad
import metacells as mc
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import distance
from sklearn.cluster import KMeans

import ambient_logger

class AmbientNoiseFinder(object):
    def __init__(
        self,
        cells_adata: ad.AnnData,
        metacells_adata: ad.AnnData,
        batches_empty_droplets_dict: dict[str, pd.Series],
        extract_batch_names_function: Callable[[ad.AnnData], list[str]],
        umi_depth_number_of_bins: int = 3,
        umi_depth_min_percentile: int = 5,
        umi_depth_max_percentile: int = 95,
        expression_delta_for_candidates_genes: int = 4,
        number_of_genes_clusters: int = 10,
        minimum_genes_in_cluster: int = 1,
        number_of_metacells_clusters: int = 10,
        minimum_metacells_in_cluster: int = 5,
        log_fractions_normalization_factor: float = 1e-5,
        genes_clusters: pd.Series = pd.Series(),
        metacells_clusters: pd.Series = pd.Series,
        remove_outliers:bool = True
    ) -> None:
        """
        Holds all the data needed to find ambient noise traces.
        This data includes the cell and metacells information, and the empty droplets information for the different batches.
        This object uses this data to identify noise-prone combinations of metacells and gene clusters, which will then be used to estimate the noise levels.

        Flow:
        1. Make sure that the genes in the cells, metacells and empty droplets information are the same - in many cases, this is not true.
        This mismatch might happen by using different versions of cell ranger, which yield different gene names,
        but also by having genes id missing in one data file while still existing on the other.
        In general, this should not happen but probably happened due to using different versions of files or pipeline processes.

        2. Extract basic information from the cells and metacells objects and add it as properties of the addata object.
        For example, the umi depth of each cell, the batch of the cell, the umi depth bin of it.

        3. Cluster the metacells and the most candidates genes.
        Here, the user can either provide the number of clusters to generate or provide the clustering data, such as gene modules and metacells annotation.

        4. For each gene cluster, rank the metacells clusters by the likelihood of having the majority of observed umis in these
        metacell-genes clusters from noise and not native expression.

        ## Data columns addad to cells addata:
        - umi_depth - the total number of umis in the cell, added to obs.
        - batch - the batch name of each cell, added to obs.
        - cells_cluster - the id of the cluster matching the metacell this cell belong to, added to bobs.
        - umi_depth_bin - the bin id matching this umi_depth, added to obs.
        - genes_cluster - the id of the gene cluster, added to var. Wil contain -1 if no candiadte gene.

        ## Data columns addad to metacell addata:
        - metacells_cluster - the id of the metacell cluster, added to obs.
        - genes_cluster - the id of the gene cluster, added to var. Wil contain -1 if no candiadte gene.

        :param cells_adata: The complete annotated data of the cells.
        :type cells_adata: ad.AnnData

        :param metacells_adata: The complete annotated data of the metacells.
        :type metacells_adata: ad.AnnData

        :param batches_empty_droplets_dict: Mapping between the batch name and the empty droplets distribution across the genes.
        :type batches_empty_droplets_dict: dict[str, pd.Series]

        :param extract_batch_names_function: Function that take the cells addata file and return the batch for each cell.
        :type extract_batch_names_function: Callable[[ad.AnnData], list[str]]

        :param umi_depth_number_of_bins: Number of bins to split the cells. This separation will produce equal number of cells in each bin based on the umi depth., defaults to 3.
        :type umi_depth_number_of_bins: int, optional

        :param umi_depth_min_percentile: Bottom percentile for cells size to remove as outliers, defaults to 5.
        :type umi_depth_min_percentile: int, optional

        :param umi_depth_max_percentile: Top percentile for cells size to remove as outliers, defaults to 95.
        :type umi_depth_max_percentile: int, optional

        :param expression_delta_for_candidates_genes: The minimal expression difference between metacells to be considered a candidate gene, defaults to 4.
        :type expression_delta_for_candidates_genes: int, optional

        :param number_of_genes_clusters: How many genes clusters should we produce, defaults to 10.
        :type number_of_genes_clusters: int, optional

        :param minimum_genes_in_cluster: Mark genes clusters with too few genes inside, which we might want to ignore in the future. defaults to 1.
        :type minimum_genes_in_cluster: int, optional

        :param number_of_metacells_clusters: How many metacells clusters should we produce, defaults to 10.
        :type number_of_metacells_clusters: int, optional

        :param minimum_metacells_in_cluster: Mark metacells clusters with too few metacells inside, which we might want to ignore in the future, defaults to 5.
        :type minimum_metacells_in_cluster: int, optional

        :param log_fractions_normalization_factor: Normalization factor to add to the fractions of cells\metacells to make sure we distinguish between 0 and 1 umi, defaults to 1e-5
        :type log_fractions_normalization_factor: float, optional

        :param genes_clusters: A series with each gene and the cluster it should be. This allows the user to provide gene modules instead of auto clusters,
                                defaults to pd.Series().
        :type genes_clusters: pd.Series, optional

        :param metacells_clusters: A series with each metacell id and the cluster it should be. This allows the user to provide annotation information instead of auto clustering,
                                    defaults to pd.Series.
        :type metacells_clusters: pd.Series, optional

        :param remove_outliers: Should we perform all the calculation only on cells which have a matching metacell or do we want to include outliers in the calculation.  defaults to True.
        :type remove_outliers: bool, optional
        """
        assert self._check_cells_metacells_empty_droplets_genes_genes(
            cells_adata, metacells_adata, batches_empty_droplets_dict
        ), "Cells, metacells and empty droplets have different gens - use utilities.remove_uncommon_genes_from_cells_metacells_empty_droplets_files() to remove those."

        self.logger = ambient_logger.logger()

        self.log_fractions_normalization_factor = log_fractions_normalization_factor
        self.umi_depth_number_of_bins = umi_depth_number_of_bins
        self.batches = list(batches_empty_droplets_dict.keys())
        self.logger.info("Working on %s batches" %(len(self.batches)))

        self.cells_adata = cells_adata[~cells_adata.obs.outlier] if remove_outliers else cells_adata
        self.metacells_adata = metacells_adata
        self.batches_empty_droplets_dict = batches_empty_droplets_dict

        self.logger.debug("Adding umi depth count to all cells")
        mc.ut.set_o_data(
            self.cells_adata,
            "umi_depth",
            pd.Series(
                data=mc.ut.get_o_numpy(self.cells_adata, name="__x__", sum=True),
                index=self.cells_adata.obs.index,
            ),
        )

        self.umi_depth_bins_thresholds = self._get_umi_depth_bin_threshold_list(
            umi_depth_number_of_bins=umi_depth_number_of_bins,
            max_percentile=umi_depth_max_percentile,
            min_percentile=umi_depth_min_percentile,
        )

        self._add_umi_depth_bins_information_to_cells_adata()
        self._add_effective_umi_depth_for_cells()
        self._add_batches_names_to_cells_adata(extract_batch_names_function)

        self.metacells_np = mc.ut.get_vo_proper(self.metacells_adata)

        self.logger.debug("Normalizing metacell information using normalize factor of %s" %log_fractions_normalization_factor)
        self.metacells_log_fractions = np.log2(
            self.metacells_np / self.metacells_np.sum(axis=1)[:, None]
            + self.log_fractions_normalization_factor
        )


        if genes_clusters.empty:
            self.logger.info("No gene clusters provided, will create %s clusters" %number_of_genes_clusters)
            genes_clusters, self.small_genes_clusters = self._get_genes_clusters(
                expression_delta_for_candidates_genes=expression_delta_for_candidates_genes,
                number_of_clusters=number_of_genes_clusters,
                minimum_genes_in_cluster=minimum_genes_in_cluster,
            )

        self._add_genes_clusters_information(genes_clusters)

        if metacells_clusters.empty:
            self.logger.info("No metacell clusters provided, will create %s clusters" % number_of_metacells_clusters)
            (
                metacells_clusters,
                self.small_metacells_clusters,
            ) = self._get_metacells_clusters(
                number_of_clusters=number_of_metacells_clusters,
                minimum_metacells_in_cluster=minimum_metacells_in_cluster,
            )

        mc.ut.set_o_data(
            self.metacells_adata, "metacells_cluster", metacells_clusters.values
        )
        self._add_cells_clusters_information()

        self.metacells_genes_clusters_median_relative_expression_to_max_df = (
            self._calculate_metacells_genes_clusters_median_relative_expression_to_max()
        )
        self.empty_droplet_genes_cluster_fraction = (
            self._get_empty_droplet_genes_cluster_fraction()
        )

        self.logger.info("Ready to estimate noise levels")

    @ambient_logger.logged()
    def _add_effective_umi_depth_for_cells(self):
        # TODO: This will be part of metacell package someday
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

    @ambient_logger.logged()
    def _add_batches_names_to_cells_adata(
        self, extract_batch_names_function: Callable[[ad.AnnData], list[str]]
    ):
        """
        Extract batch names for each cell and add this to the cells adata.

        :param extract_batch_names_function: Function that take the cells addata file and return the batch for each cell.
        :type extract_batch_names_function: Callable[[ad.AnnData], list[str]]
        """
        mc.ut.set_o_data(
            self.cells_adata, "batch", extract_batch_names_function(self.cells_adata)
        )

    @ambient_logger.logged()
    def _get_candidates_genes(
        self, expression_delta_for_candidates_genes: float
    ) -> pd.Index:
        """
        Extract all genes with enough expression difference between different metacells.
        We prefer not to use only one metacells to calculate this delta but a percentile to ensure we do not estimate the noise based on a small number of metacells or
        unique phenomena in the data.

        :param expression_delta_for_candidates_genes: The minimal expression difference between metacells to be considered a candidate gene.
        :type expression_delta_for_candidates_genes: float

        :return: A set of gene names, each considered candidate and might allow for identification of ambient noise.
        :rtype: pd.Index
        """
        self.logger.debug("Getting genes candidatesfor clustering, minimum log fold of %s" %expression_delta_for_candidates_genes)
        candidates_genes_index = np.where(
            np.percentile(self.metacells_log_fractions, 95, axis=0)
            - np.percentile(self.metacells_log_fractions, 5, axis=0)
            >= expression_delta_for_candidates_genes
        )[0]

        candidates_genes = self.metacells_adata.var.index[candidates_genes_index]
        self.logger.debug("Found %s candidates genes to cluster" %len(candidates_genes))
        return candidates_genes, candidates_genes_index

    @ambient_logger.logged()
    def _get_genes_clusters(
        self,
        expression_delta_for_candidates_genes: float,
        number_of_clusters: int,
        minimum_genes_in_cluster: int,
    ) -> tuple[pd.Series, pd.Index]:
        """
        Perform hierarchical clustering of the candidates' genes to generate genes module-like objects.
        We mark gene clusters below the required number as small_genes, which the user might be able to use or not use.
        We do not combine genes modules to make sure those clusters have enough genes to allow the user to use well-defined small genes clusters which
        will be strong enough to prevent the situation in which we combine two gene modules and harm our ability to identify genes-metacells clusters.

        :param expression_delta_for_candidates_genes: The minimal expression difference between metacells to be considered a candidate gene.
        :type expression_delta_for_candidates_genes: float

        :param number_of_clusters: How many genes clusters should we produce.
        :type number_of_clusters: int

        :param minimum_genes_in_cluster: Genes clusters with less than this number of genes will be considered small.
        :type minimum_genes_in_cluster: int

        :return: Each row is the gene's name, and the value matches the gene cluster.
        :rtype: tuple[pd.Series, pd.Index]
        """

        candidates_genes, candidates_genes_index = self._get_candidates_genes(
            expression_delta_for_candidates_genes
        )

        genes_linkeage = hierarchy.linkage(
            distance.pdist(self.metacells_log_fractions[:, candidates_genes_index].T),
            method="average",
        )

        flat_cluster = fcluster(
            genes_linkeage, t=number_of_clusters, criterion="maxclust"
        )

        clusters_series = pd.Series(
            flat_cluster.astype(int) - 1, index=candidates_genes
        )
        clusters_size = clusters_series.value_counts()

        small_genes_clusters = clusters_size.index[
            np.where(clusters_size < minimum_genes_in_cluster)[0]
        ]
        
        if len(small_genes_clusters):
            self.logger.info("%s genes clusters have low number of genes (< %d). Pass use_small_genes_clusters to AmbientNoiseEstimator if you wish to use them" % (len(small_genes_clusters), minimum_genes_in_cluster))

        for gene_cluster_id in small_genes_clusters:
            self.logger.info("Small gene cluster id: %s, genes: %s" %(gene_cluster_id, ", ".join(clusters_series[clusters_series == gene_cluster_id].index)))

        return clusters_series, small_genes_clusters

    @ambient_logger.logged()
    def _get_metacells_clusters(
        self, number_of_clusters: int, minimum_metacells_in_cluster: int
    ) -> tuple[pd.Series, pd.Index]:
        """
        Perform K-means clustering over the metacells using the candidate genes.

        :param number_of_clusters: How many metacells clusters should we produce.
        :type number_of_clusters: int

        :param minimum_metacells_in_cluster: Clusters with fewer metacells will be marked as small_metacells clusters.
        :type minimum_metacells_in_cluster: int

        :return: Each row is the id of the metacells, and the value is the cluster matching it.
        :rtype: tuple[pd.Series, pd.Index]
        """

        kmeans = KMeans(n_clusters=number_of_clusters, random_state=0, n_init=10).fit(
            self.metacells_log_fractions[
                :, np.where(self.metacells_adata.var.genes_cluster != -1)[0]
            ]
        )
        clusters_series = pd.Series(
            kmeans.labels_.astype(int), index=self.metacells_adata.obs.index.astype(int)
        )

        clusters_size = clusters_series.value_counts()

        small_metacells_clusters = clusters_size.index[
            np.where(clusters_size < minimum_metacells_in_cluster)[0]
        ]
        if len(small_metacells_clusters):
            self.logger.info("%s metacell clusters have low number of metacells (< %d). Pass use_small_metacell_clusters to AmbientNoiseEstimator if you wish to use them" % (len(small_metacells_clusters), minimum_metacells_in_cluster))

        for metacell_cluster_id in small_metacells_clusters:
            self.logger.info("Small metacell cluster id: %s, metacells: %s" %(metacell_cluster_id, ", ".join(clusters_series[clusters_series == metacell_cluster_id].index.astype(str))))

        return clusters_series, small_metacells_clusters

    def _add_cells_clusters_information(self):
        """
        For each cell in the adata object add the cluster id matching the metacell it belongs to.
        """
        mc.ut.set_o_data(
            self.cells_adata, "cells_cluster", np.full(self.cells_adata.shape[0], -1)
        )

        for i in self.metacells_adata.obs.index:
            self.cells_adata.obs.loc[
                self.cells_adata.obs.metacell == int(i), "cells_cluster"
            ] = self.metacells_adata.obs.loc[i, "metacells_cluster"]

    def _add_genes_clusters_information(self, genes_clusters: pd.Series):
        """
        For each gene in the cells and metacells adata add the cluster id to it (or -1 if not in candidates gene)

        Args:
            :param genes_clusters: Each row is the gene name and the value is the cluster id.
            :type genes_clusters: pd.Series
        """
        for adata in [self.cells_adata, self.metacells_adata]:
            mc.ut.set_v_data(adata, "genes_cluster", np.full(adata.var.shape[0], -1))
            adata.var.loc[genes_clusters.index, "genes_cluster"] = genes_clusters.values

    @ambient_logger.logged()
    def _calculate_metacells_genes_clusters_median_relative_expression_to_max(
        self,
    ) -> pd.DataFrame:
        """
        We want to find metacells-genes clusters which are more likely to be noise oriented. This means that the majority of the observed UMIs originated from ambient noise.
        To do so, we look at the expression for each combination and compare it to the maximum expression of all other metacells clusters across the same genes cluster,
        calculating the log fold.
        Metacells clusters that are LOW (negative fold) compared to other metacells clusters for the same gene cluster can be defined as
        noise-prone combinations and candidates for noise level estimation.

        :return: A 2d data frame with rows as metacells clusters, columns as genes clusters, and the value in each place is the log fold of this location vs.
        the maximum expression in the same column.
        :rtype: pd.DataFrame
        """
        genes_clusters = self.cells_adata.var.genes_cluster.max() + 1
        metacells_clusters = self.cells_adata.obs.cells_cluster.max() + 1

        expressions_df = pd.DataFrame(
            columns=range(genes_clusters), index=range(metacells_clusters), dtype=float
        )

        for gene_cluster in range(genes_clusters):
            for metacell_cluster in range(metacells_clusters):
                expressions_df.iloc[metacell_cluster, gene_cluster] = np.median(
                    self.metacells_log_fractions[
                        np.where(
                            self.metacells_adata.obs.metacells_cluster
                            == metacell_cluster
                        )[0],
                        :,
                    ][
                        :,
                        np.where(
                            self.metacells_adata.var.genes_cluster == gene_cluster
                        )[0],
                    ]
                )

        return expressions_df - expressions_df.max(axis=0)
     
    @ambient_logger.logged()
    def _get_umi_depth_bin_threshold_list(
        self,
        umi_depth_number_of_bins: int,
        min_percentile: int,
        max_percentile: int,
    ) -> list[float]:
        """
        Generate a list representing the different umi depth bins intervals based on the given cells total umi counts in this dataset.
        The user can define how many umi depth bins we need and the bottom/top percentile of cells to remove to ensure we do not use outliers.


        :param umi_depth_number_of_bins: Number of umi depth bins to split the cells into.
        :type umi_depth_number_of_bins: int

        :param min_percentile: Bottom percentile for cell size to remove as outliers.
        :type min_percentile: int

        :param max_percentile: Top percentile for cell size to remove as outliers.
        :type max_percentile: int

        :return: A umi depth threshold intervals representation: [smallest_size, 1st bin end, second bin end, ..., largest size]
        :rtype: list[int]
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

        self.logger.info("Split cells to %s bins based umi depth, provided bins: %s" %(umi_depth_number_of_bins, ["(%d,%d)" %(bins_threshold_list[i], bins_threshold_list[i+1]) for i in range(umi_depth_number_of_bins)]))
        return bins_threshold_list  # type: ignore

    @ambient_logger.logged()
    def _add_umi_depth_bins_information_to_cells_adata(self):
        """
        Go over the cells adata object and add the umi depth bin id for each cell row.
        """
        mc.ut.set_o_data(
            self.cells_adata,
            "umi_depth_bin",
            np.full(self.cells_adata.shape[0], fill_value=np.NaN),
        )

        for i in range(len(self.umi_depth_bins_thresholds) - 1):
            min_umi_depth, max_umi_depth = (
                self.umi_depth_bins_thresholds[i],
                self.umi_depth_bins_thresholds[i + 1],
            )
            self.cells_adata.obs.loc[
                (self.cells_adata.obs.umi_depth > min_umi_depth)
                & (self.cells_adata.obs.umi_depth <= max_umi_depth),
                "umi_depth_bin",
            ] = i

    @ambient_logger.logged()
    def _get_empty_droplet_genes_cluster_fraction(self) -> pd.DataFrame:
        """
        Get a 2d representation of the fraction of empty droplets per gene cluster in a batch.
        Genes clusters are independent of the metacells clusters, but they depend on the different batches we use -> this is a different ambient noise distribution based on batches.

        :return: A 2d matrix with rows as a batch, columns as gene clusters and the value in each place is the fractions of this gene cluster expression out of the whole umis in the ambient noise.
        :rtype: pd.DataFrame
        """
        genes_clusters = self.metacells_adata.var.genes_cluster.max() + 1

        empty_droplet_genes_cluster_fraction = pd.DataFrame(
            index=self.batches_empty_droplets_dict.keys(),
            columns=range(genes_clusters),
            dtype=float,
        )

        for batch_index, batch in enumerate(self.batches_empty_droplets_dict):
            batch_total_empty_droplets_umis = self.batches_empty_droplets_dict[
                batch
            ].sum()

            for genes_cluster in range(genes_clusters):
                genes = self.metacells_adata.var.genes_cluster[
                    self.metacells_adata.var.genes_cluster == genes_cluster
                ].index

                batch_empty_droplet_fraction = (
                    self.batches_empty_droplets_dict[batch].loc[genes].sum()
                    / batch_total_empty_droplets_umis
                )

                empty_droplet_genes_cluster_fraction.iloc[
                    batch_index, genes_cluster
                ] = batch_empty_droplet_fraction

        return empty_droplet_genes_cluster_fraction

    def _check_cells_metacells_empty_droplets_genes_genes(
        self,
        cells_adata: ad.AnnData,
        metacells_adata: ad.AnnData,
        batches_empty_droplets_dict: dict[str, pd.Series],
    ) -> bool:
        """
        Validate that the gene set defined in the cells, metacells and the empty droplets information is the same.

        :param cells_adata: The full cells adata obj.
        :type cells_adata: ad.AnnData

        :param metacells_adata: The full metacells adata obj.
        :type metacells_adata: ad.AnnData

        :param batches_empty_droplets_dict: A mapping between the batch name and the empty droplet series with the umi count of all the empty droplets.
        :type batches_empty_droplets_dict: dict[str, pd.Series]

        :return: Is the genes set exactly the same.
        :rtype: bool
        """
        cells_genes = cells_adata.var.index

        if len(cells_genes & metacells_adata.var.index) != len(cells_genes):
            return False

        for batch in batches_empty_droplets_dict:
            if len(batches_empty_droplets_dict[batch].index & cells_genes) != len(
                cells_genes
            ):
                return False

        return True

    def plot_expression_diff_between_clusters(
        self, show_expression_value: bool = False
    ) -> None:
        """
        Plot heatmap of the relative expression diff between metacells and gene clusters.

        :param show_expression_value: True mean that the plot will print the values of the relative expression, defaults to False
        :type show_expression_value: bool, optional
        """

        fig = plt.figure(figsize=(25, 16))

        with sb.plotting_context(rc={"font.size": 30}):
            ax = fig.add_subplot(
                111,
                xlabel="Genes clusters",
                ylabel="Metacell clusters",
                title="Metacells-genes clusters median relative expression to max",
            )

            sb.heatmap(
                self.metacells_genes_clusters_median_relative_expression_to_max_df,
                annot=show_expression_value,
                ax=ax,
            )
            _ = plt.yticks(rotation=0)
            _ = plt.xticks(rotation=0)

        plt.show()

    def plot_metacells_clustering_on_umap(self) -> None:
        """
        Plot the metacells on a umap using the metacells clustering for different colors
        """
        assert (
            "umap_x" in self.metacells_adata.obs.columns
        ), "No umap values for the metacell object"

        fig = plt.figure(figsize=(40, 20))
        ax = plt.gca()
        valid_mc_ad = self.metacells_adata[self.metacells_adata.obs.index]

        palette = sb.color_palette(
            "hls", len(self.metacells_adata.obs.metacells_cluster.value_counts())
        )

        with sb.plotting_context(rc={"font.size": 50}):
            sb.scatterplot(
                x="umap_x",
                y="umap_y",
                data=valid_mc_ad.obs,
                hue=self.metacells_adata.obs.metacells_cluster.values,
                legend="full",
                s=300,
                ax=ax,
                palette=palette,
            )
            ax.tick_params(axis="both", labelbottom=False, labelleft=False)
            ax.legend(loc="best", markerscale=3, ncol=3)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        fig.tight_layout()
        plt.show()

    def plot_genes_mc_clusters(self) -> None:
        """
        Plot a heatmap of the umis expression for the metacells and gene clusters
        """
        ordered_mc = []
        mc_colors = []
        genes_colors = []
        ordered_genes = []

        genes_clusters = self.metacells_adata.var.genes_cluster.unique()

        mc_clusters_groups = self.metacells_adata.obs.groupby("metacells_cluster")
        mc_clusters_rgb = sb.color_palette(
            "hls", len(self.metacells_adata.obs.metacells_cluster.unique())
        )
        genes_clusters_rgb = sb.color_palette("tab10", len(genes_clusters))

        for i, gene_cluster in enumerate(genes_clusters):
            if gene_cluster == -1:
                continue
            genes = self.metacells_adata.var[self.metacells_adata.var.genes_cluster == gene_cluster].index
            ordered_genes.extend(genes)
            genes_colors.extend([genes_clusters_rgb[i]] * len(genes))

        for i, mc_cluster_df in mc_clusters_groups:
            ordered_mc.extend(mc_cluster_df.index)
            mc_colors.extend([mc_clusters_rgb[i]] * mc_cluster_df.shape[0])

        genes_as_list = self.metacells_adata.var.index.to_list()
        mc_ordered_index = [int(i) for i in ordered_mc]
        genes_ordered_index = [genes_as_list.index(x) for x in ordered_genes]
        ordered_df = self.metacells_log_fractions[mc_ordered_index,:][:,genes_ordered_index]

        with sb.plotting_context(rc={"font.size": 200}):
            sb.clustermap(
                ordered_df,
                col_colors=genes_colors,
                row_colors=mc_colors,
                method="average",
                col_cluster=False,
                row_cluster=False,
                cmap="YlGnBu",
                figsize=(200, 100),
                linewidths=0,
                cbar_pos=(0.1, 0.2, 0.03, 0.5),
                yticklabels=False,
                xticklabels=False,
            )

            sb.clustermap(
                ordered_df,
                col_colors=genes_colors,
                row_colors=mc_colors,
                method="average",
                col_cluster=False,
                row_cluster=True,
                cmap="YlGnBu",
                figsize=(200, 100),
                linewidths=0,
                cbar_pos=(0.1, 0.2, 0.03, 0.5),
                yticklabels=False,
                xticklabels=False,
            )

            plt.show()

    def plot_umis_depth_bins_distribution_across_batches(self) -> None:
        """
        Plot the distribution of umi depth bins across the different batches, trying to find if there are batches with no representation of umi depth.
        """
        cells_with_umi_depth_bin = self.cells_adata[
            self.cells_adata.obs.umi_depth_bin != -1
        ]
        bin_index_count_by_batch = (
            cells_with_umi_depth_bin.obs.groupby(["batch", "umi_depth_bin"])
            .agg({"umi_depth_bin": "count"})
            .unstack(level=1)
        )

        bin_index_perc_by_batch = pd.DataFrame(
            data=bin_index_count_by_batch.to_numpy(),
            columns=range(self.umi_depth_number_of_bins),
            index=sorted(cells_with_umi_depth_bin.obs.batch.unique()),
        )

        bin_index_perc_by_batch = bin_index_perc_by_batch.div(
            bin_index_perc_by_batch.sum(axis=1), axis=0
        )

        with sb.plotting_context(rc={"font.size": 50}):
            plt.figure(figsize=(40, 20))
            ax = plt.gca()

            bin_index_perc_by_batch.plot(
                kind="bar", ax=ax, stacked=True, ylim=(0, 1), title="% bin per batch"
            )

            ax.legend(
                [
                    "bin %s: %.0f-%.0f"
                    % (
                        i,
                        self.umi_depth_bins_thresholds[i],
                        self.umi_depth_bins_thresholds[i + 1],
                    )
                    for i in range(self.umi_depth_number_of_bins)
                ],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )
            _ = plt.xticks(rotation=0)

        plt.show()

    def plot_metacells_clusters_distribution_across_batches(self) -> None:
        """
        Plot the distribution of metacells clusters across the different batches, trying to find if there are batches with no representation of cell type.
        """
        cells_with_umi_depth_bin = self.cells_adata[
            self.cells_adata.obs.umi_depth_bin != -1
        ]
        clustering_count_by_batch = (
            cells_with_umi_depth_bin.obs.groupby(["batch", "cells_cluster"])
            .agg({"cells_cluster": "count"})
            .unstack(level=1)
        )

        clustering_perc_by_batch = pd.DataFrame(
            data=clustering_count_by_batch.to_numpy(),
            columns=sorted(cells_with_umi_depth_bin.obs.cells_cluster.unique()),
            index=sorted(cells_with_umi_depth_bin.obs.batch.unique()),
        )
        clustering_perc_by_batch = clustering_perc_by_batch.div(
            clustering_perc_by_batch.sum(axis=1), axis=0
        )

        with sb.plotting_context(rc={"font.size": 50}):
            plt.figure(figsize=(40, 20))
            ax = plt.gca()

            clustering_perc_by_batch.plot(
                kind="bar",
                ylim=(0, 1),
                title="% mc clusters per batch",
                ax=ax,
                stacked=True,
                color=sb.color_palette(
                    "hls", len(self.metacells_adata.obs.metacells_cluster.unique())
                ),
            )
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            _ = plt.xticks(rotation=0)
            
        plt.show()
