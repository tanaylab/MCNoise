# +
import math
import multiprocessing as mp
import os
import pickle
import warnings
from functools import partial

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import metacells as mc
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scipy.special as scs
import seaborn as sb
import tqdm
from adjustText import adjust_text
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster, leaves_list
from scipy.spatial import distance
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split

matplotlib_axes_logger.setLevel("ERROR")

import rpy2
import rpy2.robjects as robjects
from rpy2.rinterface import RRuntimeWarning
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RRuntimeWarning)

buf = []


def f(x):
    # function that append its argument to the list 'buf'
    buf.append(x)


# output from the R console will now be appended to the list 'buf'
rpy2.rinterface_lib.callbacks.consolewrite_print = f
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = f

mpl.rcParams.update(mpl.rcParamsDefault)
sb.reset_orig()

rzetadiv = importr("zetadiv")
rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()
base = importr("base")

NON_SOLVABLE_ERR = (
    "Error in stats::lm.fit(x = x[good, , drop = FALSE] * w, y = z * w, singular.ok = FALSE,  "
    ": \n  singular fit encountered\n"
)


# -
def init_plt_params(is_clustermap=False, scale=1):
    """
    Standardize plotting parameters between graphs
    @param is_clustermap: Clustermaps need different handling of plotting
    @type is_clustermap: bool
    @param scale: Scale down or up the fonts
    @type scale: float
    """
    plt.clf()

    if is_clustermap:
        sb.reset_defaults()
        plt.rc("font", size=scale * 100)

    else:
        sb.reset_defaults()
        plt.rc("font", size=15 * scale)
        plt.rc("axes", titlesize=15 * scale)
        plt.rc("axes", labelsize=15 * scale)
        plt.rc("xtick", labelsize=15 * scale)
        plt.rc("ytick", labelsize=15 * scale)
        plt.rc("legend", fontsize=15 * scale)


# +
def get_empty_droplets_frac_from_file(
    file_info, read_file_func, empty_droplets_thr=100
):
    """
    Read one file and extract the empty droplets information from it.
    Each file suppose to be of specific batch, read it base on the read_file_func and consider all cells with less
    than empty_droplets_thr as empty.
    We make sure that there is no duplication in genes by merging the same gene together
    @param file_info: Tuple with information (file_path, batch_name)
    @type file_info: tuple(str,str)
    @param read_file_func: A function to read the file with information, for example sc.read h5ad
    @type read_file_func: func
    @param empty_droplets_thr: The max number of umis in drop to be considered as empty
    @type empty_droplets_thr: int
    @return: Dataframe with the gene name and the number of umis seen on each gene across all the empty droplets
    @rtype: pd.DataFrame
    """
    f_path, batch_name = file_info
    data = read_file_func(os.path.join(f_path))

    # find all the droplets which are considered as empty base on the threshold
    droplet_count = data.X.sum(axis=1)
    empty_droplets = data[
        np.where(np.logical_and(droplet_count > 0, droplet_count < empty_droplets_thr))[
            0
        ],
        :,
    ]
    print(
        "%s had %d(%1.f%%) out of %d empty droplets"
        % (
            batch_name,
            empty_droplets.shape[0],
            100 * empty_droplets.shape[0] / len(droplet_count),
            len(droplet_count),
        )
    )

    # agg the information from all the empty droplets
    umis_per_gene = empty_droplets.X.sum(axis=0)
    empty_droplets_df = pd.DataFrame(
        umis_per_gene.reshape(-1, 1), index=data.var.index, columns=["umis"]
    )
    empty_droplets_df["gene_n"] = empty_droplets_df.index  # combine dup genes
    empty_droplets_df = empty_droplets_df.groupby("gene_n").agg({"umis": "sum"})
    return empty_droplets_df


def get_empty_droplets_information(
    batches_to_file_path, read_file_func, empty_droplets_thr=100, num_process=np.inf
):
    """
    Extract empty droplets information from all the files, either in a multiprocess way or not
    @param batches_to_file_path: Mapping between batch name and file path with information about the batch
    @type batches_to_file_path: dict
    @param read_file_func: A function to read the file with information, for example sc.read h5ad
    @type read_file_func: func
    @param empty_droplets_thr: The max number of umis in drop to be considered as empty
    @type empty_droplets_thr: int
    @param num_process: number of processes we can run at the same time to read the files
    @type num_process: int
    @return: Mapping between the batch name and df representing all the agg information of the empty droplets
    @rtype: dict
    """
    num_processes = min(num_process, min(mp.cpu_count() - 2, len(batches_to_file_path)))
    if num_processes > 1:
        with mp.Pool(num_processes) as p:
            empty_droplets_results = p.map(
                partial(
                    get_empty_droplets_frac_from_file,
                    read_file_func=read_file_func,
                    empty_droplets_thr=empty_droplets_thr,
                ),
                [
                    (batches_to_file_path[batch], batch)
                    for batch in batches_to_file_path
                ],
            )

    else:
        empty_droplets_results = [
            read_file_func(
                batches_to_file_path[batch], empty_droplets_thr=empty_droplets_thr
            )
            for batch in batches_to_file_path
        ]

    batch_empty_droplets = {
        batch: empty_droplets_results[i] for i, batch in enumerate(batches_to_file_path)
    }
    return batch_empty_droplets


# +
# extract basic information
def get_cells_mat(cells_f_path):
    """
    Load the cells AnnData and filter out the outliers
    @param cells_f_path: path for cells anndata
    @type cells_f_path: str
    @return: AnnData obj with the non outliers cells
    @rtype: ad.AnnData
    """
    cells_ad = ad.read_h5ad(cells_f_path)
    return cells_ad[~cells_ad.obs.outlier]


# annotation functions
def annotate_based_on_df(mat_clean, annotation_df):
    """
    Add annotation column to the cells AnnData based on df with mc to cells
    @param mat_clean: The cell anndata we want to annotate
    @type mat_clean: ad.AnnData
    @param annotation_df: dataframe with mapping between mc and "type" annotation
    @type annotation_df: pd.DataFrame
    @return: cell anndata with the annotation information
    @rtype: ad.AnnData
    """
    mat_clean.obs["annotation"] = ""
    for i in annotation_df.index:
        mat_clean.obs.loc[
            mat_clean.obs.metacell == i, "annotation"
        ] = annotation_df.loc[i, "type"]

    return mat_clean


# -
def get_umi_count_threshold_list(
    cells_total_umis_count, num_bins=3, min_percentile=5, max_percentile=95
):
    """
    Split the umis count of the cells to several bins based on the percentiles.
    On default remove the low and top 5 percentiles before splitting to bins
    @param cells_total_umis_count:  UMI count per cell
    @type cells_total_umis_count: pd.Series
    @param num_bins: Amount of bins we want to divide the information
    @type num_bins: int
    @param min_percentile: minimum percentile to filter out
    @type min_percentile: int
    @param max_percentile: maximum percentile to filter out
    @type max_percentile: int
    @return: List of umi count where each adjust numbers represent bin
    @rtype: list[int]
    """
    # filter out top and low percentile to remove outliers
    total_umis_min = np.percentile(cells_total_umis_count, min_percentile)
    total_umis_max = np.percentile(cells_total_umis_count, max_percentile)
    valid_sizes = cells_total_umis_count[
        np.logical_and(
            cells_total_umis_count >= total_umis_min,
            cells_total_umis_count <= total_umis_max,
        )
    ]

    # define the different bins
    bins_threshold_list = [total_umis_min]
    for i in range(1, num_bins):
        bins_threshold_list.append(np.percentile(valid_sizes, i / num_bins * 100))

    bins_threshold_list.append(total_umis_max)

    return bins_threshold_list


# +
def convert_mc_df_to_logged_frac(metacell_df):
    """
    Move to fractions, add minimum value to make sure we separate 0 and 1 UMI in a valid way and move to log
    representation
    @param metacell_df: The metacell df with the umi count per gene and mc
    @type metacell_df: pd.DataFrame
    @return: the same data as before but in log fraction state
    @rtype: pd.DataFrame
    """
    mc_fractions = metacell_df.divide(metacell_df.sum(axis=1), axis=0)
    mc_fractions += 1e-5
    mc_fractions_logged = np.log2(mc_fractions)
    return mc_fractions_logged


def get_top_diff_genes(metacell_df, genes_min_diff=4):
    """
    Get the genes with the biggest delta between the 95 percentile and the 5 percentile across mc
    @param metacell_df: The metacell df with the umi count per gene and mc
    @type metacell_df: pd.DataFrame
    @param genes_min_diff: Minimum difference between max and min metacell to consider this top_diff gene
    @type genes_min_diff: float
    @return: list with all the top diff genes
    """
    mc_fractions_logged = convert_mc_df_to_logged_frac(metacell_df)
    top_diff_genes = mc_fractions_logged.columns[
        np.percentile(mc_fractions_logged, 95, axis=0)
        - np.percentile(mc_fractions_logged, 5, axis=0)
        >= genes_min_diff
    ]
    return top_diff_genes


def cluster_genes_and_mc(
    metacell_df,
    genes_to_use,
    num_of_genes_cluster=15,
    number_of_mc_clusters=9,
    number_mc_threshold=5,
    number_genes_threshold=1,
):
    """
    Cluster the metacells based on a list of genes
    @param metacell_df: The metacell df with the umi count per gene and mc
    @type metacell_df: pd.DataFrame
    @param genes_to_use: list of genes to use for the clustering of mc and to cluster
    @type genes_to_use: list
    @param num_of_genes_cluster: number of clusters we want for the genes
    @type num_of_genes_cluster: int
    @param number_of_mc_clusters: number of clusters we want for the mc
    @type number_of_mc_clusters: int
    @param number_mc_threshold: minimum number of mc in cluster - will alert if there is less
    @type number_mc_threshold: int
    @param number_genes_threshold: minimum number of genes in cluster - will alert if there is less
    @type number_genes_threshold: int
    @return: List with the two clusters, the mc will be a dataframe of type for each mc and the genes clusters will be
    mapped in a dict with the number of cluster g1,g2,... and the key is a list of genes
    @rtype: type(pd.DataFrame,dict)
    """
    mc_fractions_logged = convert_mc_df_to_logged_frac(metacell_df)

    # cluster the genes
    genes_linkeage = hierarchy.linkage(
        distance.pdist(mc_fractions_logged.loc[:, genes_to_use].T), method="average"
    )
    flat_cluster = fcluster(
        genes_linkeage, t=num_of_genes_cluster, criterion="maxclust"
    )
    clusters_to_genes = {"g%s" % i: [] for i in range(1, num_of_genes_cluster + 1)}
    for i, gene in enumerate(genes_to_use):
        clusters_to_genes["g%s" % flat_cluster[i]].append(gene)

    # cluster the mc
    kmeans = KMeans(n_clusters=number_of_mc_clusters, random_state=0, n_init=10).fit(
        mc_fractions_logged
    )
    #     kmeans = MiniBatchKMeans(n_clusters=number_of_mc_clusters, random_state=0, n_init=10).fit(mc_fractions_logged)
    annotation_df = pd.DataFrame({"mc": metacell_df.index, "type": kmeans.labels_})
    annotation_df["mc"] = annotation_df["mc"].astype(np.int)
    annotation_df["type"] = annotation_df["type"].astype(np.int)

    # suggest filtering based on the number of genes in cluster
    gcluster_to_remove = []
    for gcluster in clusters_to_genes:
        if len(clusters_to_genes[gcluster]) <= number_genes_threshold:
            print(
                "Gene cluster: %s has only %s gene(%s), suggest to filter it out"
                % (
                    gcluster,
                    len(clusters_to_genes[gcluster]),
                    clusters_to_genes[gcluster],
                )
            )
            gcluster_to_remove.append(gcluster)

    # suggest filtering based on the number of mc in cluster
    mc_clusters_counts = annotation_df.groupby("type").count()
    mcluster_to_remove = mc_clusters_counts.index[
        np.where(mc_clusters_counts < number_mc_threshold)[0]
    ]
    for mcluster in mcluster_to_remove:
        print(
            "mc cluster: %s has only %d mcs, suggest to filter it out"
            % (mcluster, mc_clusters_counts.loc[mcluster])
        )

    if gcluster_to_remove:
        print("forbidden_gclusters=['%s']" % "','".join(gcluster_to_remove))
    if not mcluster_to_remove.empty:
        print(
            "forbidden_mclusters=[%s]" % ",".join([str(i) for i in mcluster_to_remove])
        )

    return clusters_to_genes, annotation_df


# -


def filter_on_common_genes(mat_ad, metacells_ad, batch_empty_droplets):
    """
    Make sure every empty droplet information df, the cells anndata and the metacell anndata share the same genes,
    this is used to make sure future calculation will make sense
    @param mat_ad: The cell anndata
    @type mat_ad: ad.AnnData
    @param metacells_ad: The mc anndata we want to annotate
    @type metacells_ad: ad.AnnData
    @param batch_empty_droplets: Mapping between the batch name and df representing all the agg information of the
    empty droplets
    @type batch_empty_droplets: dict{str:pd.DataFrame}
    @return: The same input but filtered by the same genes
    @rtype: tuple(ad.AnnData, ad.AnnData, dict)
    """
    common_genes = mat_ad.var.index & metacells_ad.var.index
    for batch in batch_empty_droplets:
        common_genes = common_genes & batch_empty_droplets[batch].index

    mat_ad = mat_ad[:, common_genes]
    metacells_ad = metacells_ad[:, common_genes]
    for batch in batch_empty_droplets:
        batch_empty_droplets[batch] = batch_empty_droplets[batch].loc[common_genes]

    return mat_ad, metacells_ad, batch_empty_droplets


def get_expression_diff_df(genes_clusters, mc_clusters, metacell_df):
    """
    Calculate the diff expression between the maximum mc cluster to each other mc cluster for each gene cluster
    @param genes_clusters: mapping of the genes into clusters
    @type genes_clusters: dict{str:list}
    @param mc_clusters: dataframe of type for each mc
    @type mc_clusters: pd.DataFrame
    @param metacell_df: The metacell df with the umi count per gene and mc
    @type metacell_df: pd.DataFrame
    @return: The diff expression
    @rtype: pd.DataFrame
    """
    mc_fractions_logged = convert_mc_df_to_logged_frac(metacell_df)

    mc_clusters_groups = mc_clusters.groupby("type")
    expressions_df = pd.DataFrame(
        columns=genes_clusters.keys(),
        index=sorted(mc_clusters.type.unique()),
        dtype=np.float64,
    )
    for gcluster in genes_clusters:
        genes = genes_clusters[gcluster]
        for i, mcluster_df in mc_clusters_groups:
            expressions_df.loc[i, gcluster] = np.median(
                mc_fractions_logged.iloc[mcluster_df.mc.values, :].loc[:, genes]
            )

    expressions_diff_df = expressions_df - expressions_df.max(axis=0)
    return expressions_diff_df


# +
def get_gm_df_template(mat_ad):
    """
    Generate gene module template to be used in glm calculation later, this contains all the important
    information about the gene module
    @param mat_ad: The cell anndata
    @type mat_ad: ad.AnnData
    @return: A dataframe with all the information for about the current gene module: number of observed umis, number of total umis
    the batch name and the mc index, the fraction of the empty droplets for this mc based on the gene and the mc
    clustering annotation
    @rtype: pd.DataFrame
    """
    gm_df = pd.DataFrame(
        index=mat_ad.obs.index,
        columns=[
            "observed",
            "total_umis",
            "batch",
            "mc_ind",
            "empty_droplets_frac",
            "annotation",
        ],
    )
    gm_df["mc_ind"] = mat_ad.obs.loc[gm_df.index].metacell
    gm_df["total_umis"] = mat_ad.obs.umi_depth
    gm_df["batch"] = mat_ad.obs.loc[gm_df.index].batch
    gm_df["annotation"] = mat_ad.obs.loc[gm_df.index].annotation
    return gm_df


def get_gcluster_noisy_df(
    gcluster_id,
    mcluster_id,
    relevant_genes,
    relevant_mc,
    empty_droplet_gcluster_fraction,
    mat_df,
    current_gm_df,
):
    """
    Collect the information needed for a single noisy gene cluster and mc cluster for glm equation
    @param gcluster_id: The gene cluster id to collect information on
    @type gcluster_id: int
    @param mcluster_id:The mc cluster id to collect information on
    @type mcluster_id: int
    @param relevant_genes: List of all the genes in the current gene cluster
    @type: list
    @param relevant_mc: list of all mc in the current mc cluster
    @type relevant_mc: list
    @param empty_droplet_gcluster_fraction: mapping between batch and the empty droplets fraction information
    @type empty_droplet_gcluster_fraction: dict
    @param mat_df: dataframe with the umis count per cell in all the genes
    @type mat_df: pd.DataFrame
    @param current_gm_df: The dataframe with the relevant information about the gene module, not filtered for
    specific mc
    @type current_gm_df: pd.DataFrame
    @return: A dataframe with all the information neede for glm equation: the observed information, the expected and
    the gene module cell type (represented as Egt)
    @rtype: pd.DataFrame
    """
    current_gm_df = current_gm_df[np.in1d(current_gm_df.mc_ind, relevant_mc)]

    # add empty droplets information
    for batch in empty_droplet_gcluster_fraction.index:
        current_gm_df.loc[
            current_gm_df.batch == batch, "empty_droplets_frac"
        ] = empty_droplet_gcluster_fraction.loc[batch, gcluster_id]

    # fill in the observed umis per cell
    current_gm_df["observed"] = mat_df.loc[
        current_gm_df.index, relevant_genes
    ].values.sum(axis=1)
    current_gm_df["expected"] = (
        current_gm_df["empty_droplets_frac"] * current_gm_df["total_umis"]
    )
    current_gm_df["contamination"] = (
        current_gm_df["observed"] / current_gm_df["expected"]
    )
    current_gm_df["gmtype"] = "Egt_%s_in_%s" % (gcluster_id, mcluster_id)

    return current_gm_df


# -


def get_valid_batches_gmctypes(
    egt_df,
    batches_list,
    min_number_of_gmctypes_per_batch=10,
    min_number_of_batches_per_gmctypes=5,
):
    """
    Get a list of valid batches and gmctype by filtering those with not enough information until we are stable
    @param egt_df: A df for specific egt with all the information
    @type egt_df: pd.Data
    @param batches_list: list of ll the batches we are working on
    @type batches_list: list
    @param min_number_of_gmctypes_per_batch: Minimum number of required gmctype per batch, below that we will filter
    the batch out
    @type min_number_of_gmctypes_per_batch: int
    @param min_number_of_batches_per_gmctypes: minimum number of batches per gmctype, below this we filter the
    gmctype out
    @type min_number_of_batches_per_gmctypes: int
    @return: Egt df with valid batches and gmctypes with enough information to continue the glm calculation
    @rtype: pd.DataFrame
    """
    changed_gmctypes, changed_batches = True, True
    valid_gmctypes = list(egt_df.index)
    valid_batches = batches_list

    while changed_gmctypes or changed_batches:
        # remove gmctypes with not enough batches
        data_by_gmtype = (
            egt_df.groupby("gmtype").agg({"batch_name": "count"}).batch_name
        )
        temp_gmctypes = data_by_gmtype[
            data_by_gmtype >= min_number_of_batches_per_gmctypes
        ].index
        changed_gmctypes = len(temp_gmctypes) != len(valid_gmctypes)
        valid_gmctypes = temp_gmctypes

        # take only the valid gmctypes
        egt_df = egt_df.loc[valid_gmctypes]

        # remove batches with not enough gmctypes
        data_by_batches = egt_df.groupby("batch_name").agg({"gmtype": "count"})
        temp_batches = data_by_batches["gmtype"][
            data_by_batches["gmtype"] >= min_number_of_gmctypes_per_batch
        ].index
        changed_batches = len(temp_batches) != len(valid_batches)
        valid_batches = temp_batches
        egt_df = egt_df[egt_df.index.get_level_values(1).isin(valid_batches)]

    egt_df = egt_df.loc[valid_gmctypes]
    egt_df = egt_df[egt_df.index.get_level_values(1).isin(valid_batches)]
    return egt_df


def build_glm_equation_for_bin(df):
    """
    Convert the egt df to a glm dataframe
    @param df: the egt dataframe with all the needed information to generate the glm equation
    @type df: pd.DataFrame
    @return: A dataframe with the proper input for glm calculation
    @rtype: pd.DataFrame
    """
    num_equations = df.shape[0]
    obs_df = pd.DataFrame(df["observed"].values, columns=["obs"])

    batches_df = pd.DataFrame(
        np.repeat(df.expected.values, len(set(df.index.get_level_values(1)))).reshape(
            num_equations, len(set(df.index.get_level_values(1)))
        ),
        columns=list(set(df.index.get_level_values(1))),
    )
    egt_df = pd.DataFrame(
        np.repeat(df.total_umis.values, len(set(df.index.get_level_values(0)))).reshape(
            num_equations, len(set(df.index.get_level_values(0)))
        ),
        columns=list(set(df.index.get_level_values(0))),
    )

    # add egt columns , if the data wasn't from that egt put 0
    for egt in set(df.index.get_level_values(0)):
        egt_df.loc[df.index.get_level_values(0) == egt, egt_df.columns != egt] = 0

    # add batch information, in the data wasn't from that batch put 0
    for batch in set(df.index.get_level_values(1)):
        batches_df.loc[
            df.index.get_level_values(1) == batch, batches_df.columns != batch
        ] = 0

    df = pd.concat([obs_df, batches_df, egt_df], axis=1)
    return df[df.obs > 0]


# +
def get_valid_column_unconstraint(df):
    """
    Filter invalid columns - those with not enough information to run the glm. Sadly the current module we are using
    doesn't do it inherently so we need to run another module
    @param df: The glm dataframe
    @rtype df: pd.DataFrame
    @return: List of valid columns, those we might find solutions for
    @rtype: List
    """
    model = rzetadiv.glm_cons(
        "obs ~ . - 1",
        data=df,
        cons=1,
        family=robjects.r.poisson(link="identity"),
        method="lm.fit",
    )
    model_results = dict(zip(model.names, list(model)))
    valid_columns = list(df.columns[1:][~np.isnan(model_results["coefficients"])])
    return valid_columns


def run_model(df, number_of_tries=3):
    """
    Run the glm module to get coef for each column
    We might face RuntimeError due to invalid columns, if this is the case try to remove those columns until we quit
    @param df: the dataframe to train on
    @type df: pd.DataFrame
    @param number_of_tries: number of tries on error before stopping
    @type number_of_tries: int
    @return: The predicted coeff for each column
    @rtype: pd.DataFrame
    """
    results_df = pd.DataFrame(np.nan, index=df.columns[1:], columns=["predicted"])

    x, y = df.loc[:, df.columns[1:]], df["obs"]
    batch_columns = [i for i in x.columns if not i.startswith("Egt")]
    estart = np.array(
        [0.02] * len(batch_columns) + [1e-7] * (x.shape[1] - len(batch_columns))
    )

    try:
        model = rzetadiv.glm_fit_cons(
            x=x,
            y=y,
            cons=1,
            intercept=False,
            family=robjects.r.poisson(link="identity"),
            start=estart,
        )
        model_results = dict(zip(model.names, list(model)))
    except rpy2.rinterface_lib.embedded.RRuntimeError as ex:
        if ex.args[0] == NON_SOLVABLE_ERR:
            model_results = None
            c = 0
            while model_results is None and c < number_of_tries:
                columns_to_use = get_valid_column_unconstraint(df)
                x = df.loc[:, columns_to_use]
                batch_columns = [i for i in x.columns if not i.startswith("Egt")]
                estart = np.array(
                    [0.02] * len(batch_columns)
                    + [1e-7] * (x.shape[1] - len(batch_columns))
                )
                try:
                    model = rzetadiv.glm_fit_cons(
                        x=x,
                        y=y,
                        cons=1,
                        intercept=False,
                        family=robjects.r.poisson(link="identity"),
                        start=estart,
                    )
                    model_results = dict(zip(model.names, list(model)))
                except Exception as ex:
                    print(ex)
                    c += 1

            if model_results is None:
                print(
                    "Tried fitting the model 3 times and failed, moving to diffeent fold"
                )
                return None, None

        else:
            print(ex)
            return None, None

    results_df.loc[x.columns, "predicted"] = model_results["coefficients"]
    return results_df


def predict_alphas_and_egt(df, num_cv=10):
    """
    Solve a Poisson glm model to get the coefficient of alphas per batch and E_gt per gene module and cell type cluster
    Get a df (output of build_glm_df) where each row holds the results of specific gene modules + type and batch
    Columns should be obs (for the observed umis) and the rest of the columns is the expected noisy umis for this row
    (if alpha=1) and E_gt for this row
    @param df: A dataframe with the glm equations to solve
    @type df: pd.DataFrame
    @param num_cv: Number of cross validation to run
    @type num_cv: int
    @return:
        - Alphas: fractions of noise (should be multiply by 100 to get percentage)
        - E_gt: fractions of umis
    @rtype: pd.DataFrame
    """

    # start to solve without constraint, maybe some of the columns need to be removed
    results = []
    df = df.astype(np.float64)

    valid_columns = ["obs"] + get_valid_column_unconstraint(df)
    df = df.loc[:, valid_columns]

    if num_cv == 1:
        results.append(run_model(df))

    else:
        for i in range(num_cv):
            train, _ = train_test_split(df, test_size=0.1, random_state=i)
            results.append(run_model(train))

    return pd.DataFrame(
        {
            "predicted": np.nanmean(results, axis=0).reshape(-1),
            "predicted_sd": np.nanstd(results, axis=0).reshape(-1),
        },
        index=results[0].index,
    )


# -


def split_cells_based_on_bin(egt_df, bins_threshold_list, max_observed_threshold=None):
    """
    Aggregate all the cells of a specific gm and of specific type based to bins
    Using observed threshold to remove cells with too many umis, this is based on the distribution of the noisy cells
    @param egt_df: Dataframe with the egt information for all the umis range
    @type egt_df: pd.DataFrame
    @param bins_threshold_list: List of umi count where each adjust numbers represent bin
    @type bins_threshold_list: list
    @param max_observed_threshold: Maximum numbers of observed umis to make us think that this isn't noise, if none is
    given calculate it by 4 times the median
    @type max_observed_threshold: int
    @return: Mapping between each bin and the data frame with the information about it
    @rtype: dict
    """
    agg_df_list = {}

    if max_observed_threshold is None:
        max_observed_threshold = (egt_df.observed.median() + 1) * 4

    egt_df = egt_df[egt_df.observed <= max_observed_threshold]

    for i in range(1, len(bins_threshold_list)):
        min_value, max_value = bins_threshold_list[i - 1], bins_threshold_list[i]
        bin_df = (
            egt_df[
                np.logical_and(
                    egt_df.total_umis > min_value, egt_df.total_umis <= max_value
                )
            ]
            .groupby("batch")
            .agg({"observed": "sum", "expected": "sum", "total_umis": "sum"})
        )
        bin_df["contamination"] = bin_df["observed"] / bin_df["expected"]
        bin_df["bin_name"] = "%.0f < s <= %.0f" % (min_value, max_value)
        bin_df["bin"] = i
        bin_df["gmtype"] = egt_df["gmtype"][0]
        agg_df_list[i] = bin_df

    return agg_df_list


# +


def get_ticks_based_on_value(max_value):
    """
    Get ticks range which make sense based on the max value
    @param max_value: The max value of the ticks
    @type max_value: float
    @return: list of ticks for plots
    @rtype: list
    """
    if max_value > 10:
        ticks = np.arange(0, max_value, 2)
    elif max_value > 5:
        ticks = np.arange(0, max_value, 1)
    else:
        ticks = [round(i, 1) for i in np.arange(0, max_value * 1.1, max_value / 10)]

    return ticks


def plot_noise_egt_estimation_over_thresholds(
    calculation_results,
    show_lineplot=True,
    show_heatmap=False,
    legend_ncols_batches=3,
    legend_ncols_egt=5,
):
    """
    Plot the egt estimation on different thresholds based on the glm model
    @param calculation_results: A mapping between the bin and all the glm results of this bin
    @type calculation_results: dict{int:pd.DataFrame}
    @param show_lineplot: Should we plot lineplot
    @type show_lineplot: bool
    @param show_heatmap: Should we plot heatmap
    @type show_heatmap: bool
    @param legend_ncols_batches:
    @type legend_ncols_batches: number of columns for batches
    @param legend_ncols_egt:
    @type legend_ncols_egt: number of columns for egt
    """
    assert (
        show_lineplot or show_heatmap
    ), "At least show_lineplot or show_heatmap should be true"

    batches_info_combined, egt_info_combined = {}, {}
    max_batch_value, max_egt_value = 0, 0

    # Combine all the information and find the max value for plot ticks
    glm_combined = {
        bin_size: pd.concat(calculation_results[bin_size])
        for bin_size in calculation_results
    }
    for bin_size in glm_combined:
        batches_info, egt_info = split_results_to_batches_egt(glm_combined[bin_size])
        batches_info_combined[bin_size] = batches_info
        egt_info_combined[bin_size] = egt_info

        max_batch_value = max(max_batch_value, np.max(batches_info.predicted))
        max_egt_value = max(max_egt_value, np.max(egt_info.predicted))

    if show_lineplot:
        yticks = get_ticks_based_on_value(max_batch_value)

        # batches alpha
        init_plt_params(scale=3)
        for bin_size in calculation_results:

            batches_info, egt_info = (
                batches_info_combined[bin_size],
                egt_info_combined[bin_size],
            )
            colors = sb.color_palette("hls", len(batches_info.index.unique()))

            fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(
                111,
                xlabel="threshold",
                ylabel="Predicted alpha (%)",
                title="Predicted alpha over thresholds - bin %s" % bin_size,
                ylim=(0, max_batch_value * 1.1),
                xlim=(
                    glm_combined[bin_size].threshold.min(),
                    glm_combined[bin_size].threshold.max(),
                ),
                yticks=yticks,
            )

            for i, batch in enumerate(sorted(batches_info.index.unique())):
                current_batch = batches_info.loc[batch]
                if current_batch.empty:
                    continue

                ax.errorbar(
                    x=current_batch.threshold,
                    y=current_batch.predicted,
                    yerr=current_batch.predicted_sd,
                    c=colors[i],
                    capsize=5,
                    label=batch,
                    elinewidth=5,
                    lw=5,
                )

            plt.grid(axis="y")
            plt.legend(
                bbox_to_anchor=(1.04, 1), loc="upper left", ncol=legend_ncols_batches
            )
            plt.show()

        # egt
        init_plt_params(scale=3)
        for bin_size in calculation_results:
            batches_info, egt_info = (
                batches_info_combined[bin_size],
                egt_info_combined[bin_size],
            )
            colors = sb.color_palette("hls", len(egt_info.index.unique()))

            fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(
                111,
                xlabel="threshold",
                ylabel="Predicted Egt fraction",
                title="Predicted Egt values over thresholds - bin %s" % bin_size,
                ylim=(0, max_egt_value * 1.1),
                xlim=(
                    glm_combined[bin_size].threshold.min(),
                    glm_combined[bin_size].threshold.max(),
                ),
                yticks=np.linspace(0, max_egt_value * 1.1, num=10),
            )

            for i, egt in enumerate(sorted(egt_info.index.unique())):
                current_egt = egt_info.loc[egt]
                ax.errorbar(
                    x=current_egt.threshold,
                    y=current_egt.predicted,
                    yerr=current_egt.predicted_sd,
                    c=colors[i],
                    capsize=5,
                    label=egt,
                    elinewidth=5,
                    lw=5,
                )

            plt.grid(axis="y")
            plt.yscale("symlog", linthreshy=1e-5, basey=10)
            #             plt.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            plt.legend(
                bbox_to_anchor=(1.04, 1), loc="upper left", ncol=legend_ncols_egt
            )
            plt.show()

    if show_heatmap:
        sb.reset_defaults()
        for bin_size in calculation_results:
            batches_info, _ = (
                batches_info_combined[bin_size],
                egt_info_combined[bin_size],
            )

            data = batches_info.pivot("name", "threshold", "predicted")
            fig, ax = plt.subplots(figsize=(8, 8))
            sb.heatmap(
                data,
                xticklabels=["%.1f" % i for i in data.columns],
                vmin=0,
                vmax=max_batch_value,
                cmap="YlGnBu",
                ax=ax,
            )
            plt.title("Predicted alpha over thresholds - bin %s" % bin_size)
            plt.show()

        for bin_size in calculation_results:
            _, egt_info = batches_info_combined[bin_size], egt_info_combined[bin_size]
            fig, ax = plt.subplots(figsize=(8, 16))
            data = egt_info.pivot("name", "threshold", "predicted")
            sb.heatmap(
                data,
                xticklabels=["%.1f" % i for i in data.columns],
                vmin=0,
                vmax=max_egt_value,
                cmap="YlGnBu",
                ax=ax,
            )
            plt.title("Predicted Egt values over thresholds - bin %s" % bin_size)
            plt.show()


def plot_batch_egt_tracker(batches_egt_tracker):
    """
    Plot the amount of batches and egt gained in each threshold
    @param batches_egt_tracker: Mapping between teh threshold and the amount of batches and egt used in it
    @type batches_egt_tracker: dict
    """
    init_plt_params()
    for i, bin_size in enumerate(batches_egt_tracker):
        batches_egt_tracker[bin_size].plot(
            title=r"# batches\egt per threshold - bin %s" % bin_size,
            grid=True,
            xlabel="threshold",
            ylabel="#count",
            figsize=(5, 4),
        )

    plt.tight_layout()
    plt.show()


def plot_batches_predictions_across_bins_as_barplot(batches_prediction, batches):
    """
    Plot the alphas of all the bins as one barplot
    @param batches_prediction: The dataframe with prediction of all the bins
    @type batches_prediction: pd.DataFrame
    @param batches: list of batches to plot, will define the order
    @type batches: list
    """
    init_plt_params()
    batches_prediction.bin = batches_prediction.bin.astype(np.int)
    bins_list = sorted(batches_prediction.bin.unique())

    colors = sb.color_palette("tab10", len(bins_list))
    colors_dict = {i: colors.pop() for i in bins_list}

    xticks = np.arange(len(batches) * len(bins_list), step=len(bins_list))
    yticks = get_ticks_based_on_value(
        int(np.ceil(np.max(batches_prediction["predicted"]))) + 1
    )

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(
        111,
        xlabel="Batch",
        ylabel="Predicted alpha",
        title="Batches alphas by bin",
        xticks=xticks,
        yticks=yticks,
    )

    for p, bin_size in enumerate(bins_list):
        batch_df = batches_prediction[batches_prediction.bin == bin_size]
        predicted_l, predicted_sd_l = [], []

        for batch in batches:
            if batch in batch_df.index:
                predicted_l.append(batch_df.loc[batch].predicted)
                predicted_sd_l.append(batch_df.loc[batch].predicted_sd)
            else:
                predicted_l.append(np.nan)
                predicted_sd_l.append(np.nan)

        ax.bar(
            xticks + p * 0.5,
            predicted_l,
            yerr=predicted_sd_l,
            width=0.5,
            label=bin_size,
            color=colors_dict[bin_size],
            error_kw=dict(lw=1, capsize=3, capthick=1),
        )

    ax.set_xticklabels(batches, rotation="vertical")
    ax.grid()
    plt.legend()
    plt.show()


def plot_batches_predictions_across_bins(
    batches_prediction, show_txt=False, show_err=False, font_scale=1.5
):
    """
    Plot the alphas of all the bins as one barplot between different bins
    @param batches_prediction: The dataframe with prediction of all the bins
    @type batches_prediction: pd.DataFrame
    @param show_txt: Should we show the name of the batch next to it
    @type show_txt: bool
    @param show_err: Should we show error bars next to prediction
    @type show_err: bool
    @param font_scale: Scaling the texts and fonts
    @type font_scale: float
    """
    # Choose the maximum bin as the common x - axis
    x_bin = batches_prediction.bin.max()
    x_df = batches_prediction[batches_prediction.bin == x_bin]
    max_batches = int(np.ceil(batches_prediction.predicted.max()))

    init_plt_params(scale=font_scale)

    for bin_size in batches_prediction.bin.unique():
        if bin_size == x_bin:
            continue

        texts = []
        y_df = batches_prediction[batches_prediction.bin == bin_size]

        fig = plt.figure()
        ax = fig.add_subplot(
            111,
            xlabel="Alpha for bin %s" % x_bin,
            ylabel="Alpha for bin %s" % bin_size,
            title="Predicted alphas bin %s vs %s" % (bin_size, x_bin),
            xticks=get_ticks_based_on_value(max_batches),
            yticks=get_ticks_based_on_value(max_batches),
        )

        shared_batches = x_df.index & y_df.index

        if show_err:
            plt.errorbar(
                x=x_df.loc[shared_batches, "predicted"],
                y=y_df.loc[shared_batches, "predicted"],
                xerr=x_df.loc[shared_batches, "predicted_sd"],
                yerr=y_df.loc[shared_batches, "predicted_sd"],
                fmt=".",
                ecolor="red",
                c="black",
                ms=8,
            )
        else:
            plt.scatter(
                x=x_df.loc[shared_batches, "predicted"],
                y=y_df.loc[shared_batches, "predicted"],
                s=10,
            )

        if show_txt:
            for _, txt in enumerate(shared_batches):
                t = plt.text(
                    x=x_df.loc[txt, "predicted"], y=y_df.loc[txt, "predicted"], s=txt
                )
                texts.append(t)

        plt.plot([0, max_batches], [0, max_batches], "--k")

        if show_txt:
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle="-", color="red", lw=0.5),
                expand_text=(1, 1),
            )

        plt.show()


def plot_expression_diff_between_clusters(
    genes_clusters, mc_clusters, metacell_df, scale=2
):
    """
    Plot heatmap of the expression diff between mc and gene clusters
    @param genes_clusters: mapping of the genes into clusters
    @type genes_clusters: dict{str:list}
    @param mc_clusters: dataframe of type for each mc
    @type mc_clusters: pd.DataFrame
    @param metacell_df: The metacell df with the umi count per gene and mc
    @type metacell_df: pd.DataFrame
    """
    expressions_diff_df = get_expression_diff_df(
        genes_clusters, mc_clusters, metacell_df
    )

    init_plt_params(scale=scale)

    fig = plt.figure(figsize=(25, 16))
    ax = fig.add_subplot(
        111,
        xlabel="Genes clusters",
        ylabel="mc clusters",
        title="Differential expression between mc clusters and max expression for each gene cluster",
    )

    g = sb.heatmap(expressions_diff_df, annot=True, ax=ax)
    _ = plt.yticks(rotation=0)
    _ = plt.xticks(rotation=0)

    plt.show()


# -
def split_results_to_batches_egt(results_df):
    """
    Split the egt and alpha prediction results into different dataframes, one for egt and one for batches alpha
    @param results_df: The results dataframe with coeff for all batches and egt
    @type results_df: pd.DataFrame
    @return: Two dataframes, one for batches coeff and one for egt coeff
    @rtype: tuple(pd.DataFrame, pd.DataFrame)
    """
    batches_info = results_df[~results_df.index.str.startswith("Egt")]
    batches_info["predicted"] = batches_info["predicted"] * 100
    batches_info["predicted_sd"] = batches_info["predicted_sd"] * 100

    if "bin" in batches_info.columns:
        batches_info.bin = batches_info.bin.astype(np.int)

    egt_info = results_df[results_df.index.str.startswith("Egt")]
    if "bin" in egt_info.columns:
        egt_info = egt_info.drop("bin", axis=1)

    return batches_info, egt_info


def plot_batch_egt_results_on_single_plot(
    glm_result,
    bins_to_compare=[],
    threshold=-2,
    show_txt=False,
    batches_to_plot=[],
    egt_to_plot=[],
    show_batches=True,
    show_egt=True,
    show_err=False,
    scale=2,
):
    """
    Plot batches and Egt comparison between thresholds on a specific threshold - all bins together
    @param glm_result: The batches and egt coeff
    @type glm_result: pd.DataFrame
    @param bins_to_compare: which bins to compare, if non we will comnpare all
    @type bins_to_compare: list
    @param threshold: threshold to compare on
    @type threshold: float
    @param show_txt: should we show the name of the batch next to it
    @type show_txt: bool
    @param batches_to_plot: which batches we should plot, all if not provided
    @type batches_to_plot: list[str]
    @param egt_to_plot: which egt we should plot, all if not provided
    @type egt_to_plot: list[str]
    @param show_batches: should we plot batch information
    @type show_batches: bool
    @param show_egt: should we plot egt information
    @type show_egt: bool
    @param show_err: should we show error bars
    @type show_err: bool
    """
    assert len(glm_result) > 1, "Can't compare egt with 1 bin"

    # inner function to plot comparison
    def _plot_graph(
        info_dict,
        white_list,
        colors,
        max_bin,
        show_err=False,
        show_txt=False,
        is_egt=False,
    ):
        init_plt_params(scale=scale)
        fig, ax = plt.subplots()
        max_v = 0
        texts = []
        for i, bin_size in enumerate(bins_to_compare):
            shared = info_dict[max_bin].index & info_dict[bin_size].index
            shared = np.intersect1d(shared, white_list) if len(white_list) else shared

            max_v = max(
                max_v,
                max(
                    np.max(info_dict[bin_size].loc[shared, "predicted"]),
                    np.max(info_dict[max_bin].loc[shared, "predicted"]),
                ),
            )

            if show_err:
                plt.errorbar(
                    x=info_dict[max_bin].loc[shared, "predicted"],
                    y=info_dict[bin_size].loc[shared, "predicted"],
                    yerr=info_dict[max_bin].loc[shared, "predicted_sd"],
                    xerr=info_dict[bin_size].loc[shared, "predicted_sd"],
                    c=colors[i],
                    label="bin %s" % bin_size,
                    fmt=".",
                    ms=8,
                )
            else:
                plt.errorbar(
                    x=info_dict[max_bin].loc[shared, "predicted"],
                    y=info_dict[bin_size].loc[shared, "predicted"],
                    fmt=".",
                    c=colors[i],
                    label="bin %s" % bin_size,
                    ms=8,
                )

            if show_txt:
                for _, txt in enumerate(shared):
                    t = plt.text(
                        x=info_dict[max_bin].loc[txt, "predicted"],
                        y=info_dict[bin_size].loc[txt, "predicted"],
                        s=txt,
                        fontsize=8,
                    )
                    texts.append(t)

        plt.legend()

        max_v = max_v if is_egt else math.ceil(max_v)
        label = "alphas" if not is_egt else "egt"

        plt.plot([0, max_v], [0, max_v], "--k")
        plt.title("Predicted %s on for threshold: %s" % (label, threshold))
        plt.xlabel("Predicted on bin %s" % max_bin)
        plt.ylabel("Predicted on different bins")
        if show_txt:
            adjust_text(
                texts,
                force_points=0.2,
                force_text=0.2,
                expand_points=(1, 1),
                expand_text=(1, 1),
                arrowprops=dict(arrowstyle="-", color="red", lw=0.5),
            )

        if is_egt:
            plt.yscale("symlog", linthreshy=1e-5, basey=10)
            plt.xscale("symlog", linthreshx=1e-5, basex=10)
        else:
            plt.xticks(range(0, max_v))
            plt.yticks(range(0, max_v))

        plt.show()

    # Find the closest threshold as requested and collet the information
    if not len(bins_to_compare):
        bins_to_compare = list(glm_result)[:-1]

    batches_info_dict, egt_info_dict = {}, {}
    max_bin = max(glm_result)

    absolute_difference_function = lambda list_value: abs(list_value - threshold)
    colors = sb.color_palette("hls", len(bins_to_compare))

    for bin_size in glm_result:
        batches_info, egt_info = split_results_to_batches_egt(
            pd.concat(glm_result[bin_size])
        )
        closest_value = min(
            batches_info.threshold.unique(), key=absolute_difference_function
        )
        batches_info_dict[bin_size] = batches_info[
            batches_info.threshold == closest_value
        ]
        egt_info_dict[bin_size] = egt_info[egt_info.threshold == closest_value]

    # Actually plot the graphs
    if show_batches:
        _plot_graph(
            batches_info_dict,
            batches_to_plot,
            colors,
            max_bin=max_bin,
            show_err=show_err,
            show_txt=show_txt,
            is_egt=False,
        )

    if show_egt:
        _plot_graph(
            egt_info_dict,
            egt_to_plot,
            colors,
            max_bin=max_bin,
            show_err=show_err,
            show_txt=show_txt,
            is_egt=True,
        )


def plot_egt_comparison_on_single_threshold(
    calculation_results, threshold, egt_to_show=[], scale=2
):
    """
    Plot the egt results of specific threshold of all the bins vs the maximum bin
    @param calculation_results:  The batches and egt coeff
    @type calculation_results: pd.DataFrame
    @param threshold: The threshold to plot
    @type threshold: float
    @param egt_to_show: The egt to show, all if not given
    @type egt_to_show: list[str]
    """
    assert len(calculation_results) > 1, "Can't compare egt with 1 bin"

    x_bin = list(calculation_results)[-1]
    absolute_difference_function = lambda list_value: abs(list_value - threshold)
    egt_info_dict = {}

    max_egt = 0
    min_egt = 0

    for bin_size in calculation_results:
        _, egt_info = split_results_to_batches_egt(
            pd.concat(calculation_results[bin_size])
        )
        closest_value = min(
            egt_info.threshold.unique(), key=absolute_difference_function
        )
        egt_info_dict[bin_size] = egt_info[egt_info.threshold == closest_value]

        max_egt = max(
            max_egt, np.max(egt_info[egt_info.threshold == closest_value].predicted)
        )
        min_egt = min(
            min_egt, np.min(egt_info[egt_info.threshold == closest_value].predicted)
        )

    init_plt_params(scale=scale)

    for bin_size in list(calculation_results)[:-1]:
        share_egt = egt_info_dict[x_bin].index & egt_info_dict[bin_size].index
        share_egt = (
            np.intersect1d(share_egt, egt_to_show) if len(egt_to_show) else share_egt
        )

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(
            111,
            xlabel="Predicted on bin %s" % x_bin,
            ylabel="Predicted on bin %s" % bin_size,
            title="Egt bin %s vs bin %s" % (bin_size, x_bin),
        )
        plt.scatter(
            x=egt_info_dict[x_bin].loc[share_egt, "predicted"],
            y=egt_info_dict[bin_size].loc[share_egt, "predicted"],
            s=40,
        )
        plt.plot([min_egt, max_egt], [min_egt, max_egt], "--k")

        plt.grid()
        plt.yscale("symlog", linthreshy=1e-5, basey=10)
        plt.xscale("symlog", linthreshx=1e-5, basex=10)

        plt.show()


def plot_alphas_predictions_before_after_egt_share(
    pre_share_results, post_share_results, threshold, scale=2
):
    """
    Plot the prediction of batches coeff change before sharing egt
    @param pre_share_results: The glm results were each bin used it's own egt prediction
    @type pre_share_results: dict
    @param post_share_results: Dataframe with the final results were all the bins use the same egt
    @type post_share_results: pd.DataFrame
    @param threshold: the threshold on the pre share results to compare to
    @type threshold: float
    """
    post_share_batches_prediction, _ = split_results_to_batches_egt(post_share_results)

    # find the closest threshold and gather plotting information for axis
    absolute_difference_function = lambda list_value: abs(list_value - threshold)
    max_alpha = 0
    for bin_size in pre_share_results:
        pre_batches_info, _ = split_results_to_batches_egt(
            pd.concat(pre_share_results[bin_size])
        )
        closest_value = min(
            pre_batches_info.threshold.unique(), key=absolute_difference_function
        )
        pre_batches_info = pre_batches_info[pre_batches_info.threshold == closest_value]
        shared = np.intersect1d(
            pre_batches_info.index,
            post_share_batches_prediction[
                post_share_batches_prediction.bin == bin_size
            ].index,
        )

        max_alpha = max(
            max(
                np.max(pre_batches_info.loc[shared].predicted),
                np.max(
                    post_share_batches_prediction[
                        post_share_batches_prediction.bin == bin_size
                    ]
                    .loc[shared]
                    .predicted
                ),
            ),
            max_alpha,
        )

    # plot for each bin
    init_plt_params(scale=scale)
    g = []
    for bin_size in pre_share_results:
        pre_batches_info, _ = split_results_to_batches_egt(
            pd.concat(pre_share_results[bin_size])
        )
        closest_value = min(
            pre_batches_info.threshold.unique(), key=absolute_difference_function
        )
        pre_batches_info = pre_batches_info[pre_batches_info.threshold == closest_value]
        shared = np.intersect1d(
            pre_batches_info.index,
            post_share_batches_prediction[
                post_share_batches_prediction.bin == bin_size
            ].index,
        )

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(
            111,
            xlabel="Before sharing",
            ylabel="After sharing",
            title="alpha with Egt sharing (bin %s)" % bin_size,
            xticks=get_ticks_based_on_value(max_alpha),
            yticks=get_ticks_based_on_value(max_alpha),
        )

        plt.scatter(
            x=pre_batches_info.loc[shared].predicted,
            y=post_share_batches_prediction[
                post_share_batches_prediction.bin == bin_size
            ]
            .loc[shared]
            .predicted,
            s=100,
        )

        plt.grid()
        plt.plot([0, max_alpha], [0, max_alpha], "--k")
        plt.tight_layout()
        plt.show()
        g.append(fig)


# +
def plot_annotation_on_umap(metacells_ad, annotation_df):
    """
    Plot the annotation of metacells on umap with annotation for colors
    @param metacells_ad: The mc anndata we want to annotate
    @type metacells_ad: ad.AnnData
    @param annotation_df: dataframe with annotation information per mc
    @type annotation_df: pd.DataFrame
    """
    init_plt_params(scale=4)

    fig = plt.figure(figsize=(40, 20))
    ax = plt.gca()
    valid_mc_ad = metacells_ad[annotation_df.index]

    palette = (
        annotation_df[["type", "color"]].set_index("type").to_dict()["color"]
        if "color" in annotation_df.columns
        else sb.color_palette("hls", len(annotation_df.type.value_counts()))
    )
    sb.scatterplot(
        x="umap_x",
        y="umap_y",
        data=valid_mc_ad.obs,
        hue=annotation_df["type"].values,
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


def plot_genes_mc_clusters(
    genes_clusters, mc_clusters, metacell_df, scale=1, show_tickslabels=True
):
    """
    Plot heatmap of the expression for mc and gene clusters
    @param genes_clusters: mapping of the genes into clusters
    @type genes_clusters: dict{str:list}
    @param mc_clusters: dataframe of type for each mc
    @type mc_clusters: pd.DataFrame
    @param metacell_df: The metacell df with the umi count per gene and mc
    @type metacell_df: pd.DataFrame
    """
    ordered_mc = []
    mc_colors = []
    genes_colors = []
    ordered_genes = []

    mc_fractions_logged = convert_mc_df_to_logged_frac(metacell_df)

    mc_clusters_groups = mc_clusters.groupby("type")
    mc_clusters_rgb = sb.color_palette("hls", len(mc_clusters.type.unique()))
    genes_clusters_rgb = sb.color_palette("tab10", len(genes_clusters))

    for i, k in enumerate(genes_clusters):
        ordered_genes.extend(genes_clusters[k])
        genes_colors.extend([genes_clusters_rgb[i]] * len(genes_clusters[k]))

    for i, mc_cluster_df in mc_clusters_groups:
        ordered_mc.extend(mc_cluster_df.mc.values)
        mc_colors.extend([mc_clusters_rgb[i]] * mc_cluster_df.shape[0])

    ordered_df = mc_fractions_logged.iloc[ordered_mc, :].loc[:, ordered_genes]

    init_plt_params(is_clustermap=True, scale=scale)
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
        yticklabels=show_tickslabels,
        xticklabels=show_tickslabels,
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
        yticklabels=show_tickslabels,
        xticklabels=show_tickslabels,
    )

    plt.show()


# -


def get_shared_ctypes(calculation_results, threshold_to_egt_dict, threshold=-4):
    """
    Get the shared ctype between different thresholds
    @param calculation_results: glm results for all bins and thresholds
    @type calculation_results: dict
    @param threshold_to_egt_dict: mapping between the thresholds and and egt in that threshold
    @type threshold_to_egt_dict: dict
    @param threshold: Maximum number of differential expression of gmctype to use
    @type threshold: float
    @return:
    """
    valid_gmctype = threshold_to_egt_dict[threshold]
    absolute_difference_function = lambda list_value: abs(list_value - threshold)
    valid_gmctype_formatted = [i.replace(" ", "_") for i in valid_gmctype]

    shared_egt = None

    for bin_s in calculation_results:
        _, egt_info = split_results_to_batches_egt(
            pd.concat(calculation_results[bin_s])
        )
        closest_value = min(
            egt_info.threshold.unique(), key=absolute_difference_function
        )
        egt_info = egt_info[egt_info.threshold == closest_value]
        shared_egt_temp = np.intersect1d(valid_gmctype_formatted, egt_info.index)
        shared_egt = (
            np.intersect1d(shared_egt_temp, shared_egt)
            if shared_egt
            else shared_egt_temp
        )

        return shared_egt


# +
def calculate_alpha_egt_single_bin(
    bin_size,
    egt_per_bin_df_dict,
    threshold_to_egt_dict,
    batches,
    min_number_of_gmctypes_per_batch=10,
    min_number_of_batches_per_gmctypes=5,
    min_expected_umi_threshold=100,
):
    """
    Run glm for a single bin and return the results of the prediction
    @param bin_size: The bin to calculate the glm on
    @type bin_size: int
    @param egt_per_bin_df_dict: Mapping between bin and the egt dataframe of it
    @type egt_per_bin_df_dict: dict
    @param threshold_to_egt_dict: mapping between threshold (diff exp) to egt valid in this threshold
    @type threshold_to_egt_dict: dict
    @param batches: list of batches
    @type batches: list
    @param min_number_of_gmctypes_per_batch: Minimum number of required gmctype per batch, below that we will filter
    the batch out
    @type min_number_of_gmctypes_per_batch: int
    @param min_number_of_batches_per_gmctypes: minimum number of batches per gmctype, below this we filter the
    gmctype out
    @type min_number_of_batches_per_gmctypes: int
    @param min_expected_umi_threshold: Minimum number of expected umis per gm to make this a valid gm
    @type min_expected_umi_threshold: int
    @return: Tuple of the bin size, the prediction results and tracker of the number of batches and egt per threshold
    @rtype: tuple
    """
    results = []
    num_gt_l, num_batches_l, valid_batches, valid_gmctype = [], [], [], []

    # filtered out data with not enough expected umis and collect the data from all the egt
    data_combined = pd.concat(egt_per_bin_df_dict[bin_size])
    data_combined = data_combined[data_combined.expected >= min_expected_umi_threshold]
    data_combined["batch_name"] = data_combined.index.get_level_values(1)
    glm_equations = build_glm_equation_for_bin(data_combined)
    egt_with_enough_info = data_combined.index.get_level_values(0).unique()

    # run over thresholds and predict the coeff for the egt and batches
    for thr in tqdm.tqdm(
        threshold_to_egt_dict,
        desc="Fitting over different thresholds for bin: %s" % bin_size,
    ):
        egt_to_use = egt_with_enough_info & threshold_to_egt_dict[thr]

        # no new info to work with
        if (
            len(egt_to_use) == len(valid_gmctype)
            or len(egt_to_use) < min_number_of_gmctypes_per_batch
        ):
            num_batches_l.append(len(valid_batches))
            num_gt_l.append(len(valid_gmctype))
            continue

        valid_egt = get_valid_batches_gmctypes(
            data_combined.loc[egt_to_use],
            batches,
            min_number_of_gmctypes_per_batch=min_number_of_gmctypes_per_batch,
            min_number_of_batches_per_gmctypes=min_number_of_batches_per_gmctypes,
        )

        if valid_egt.empty:
            num_batches_l.append(0)
            num_gt_l.append(0)
            continue

        else:
            valid_gmctype = valid_egt.index.get_level_values(0).unique()
            valid_batches = valid_egt.index.get_level_values(1).unique()
            num_gt_l.append(len(valid_gmctype))
            num_batches_l.append(len(valid_batches))

        current_equations = get_filtered_eq(glm_equations, valid_batches, valid_gmctype)
        glm_results = predict_alphas_and_egt(current_equations, num_cv=10)

        glm_results["threshold"] = thr
        glm_results["name"] = glm_results.index
        results.append(glm_results)

    batches_egt_tracker = pd.DataFrame(
        np.array([num_batches_l, num_gt_l]).T,
        index=threshold_to_egt_dict.keys(),
        columns=["num_batches", "num_gt"],
    )

    return bin_size, results, batches_egt_tracker


def calculate_alpha_egt(
    egt_per_bin_df_dict,
    threshold_to_egt_dict,
    batches,
    num_bins,
    min_number_of_gmctypes_per_batch=10,
    min_number_of_batches_per_gmctypes=5,
    min_expected_umi_threshold=100,
    run_mp=False,
):
    """
    Run glm for all the bins and return the results of the prediction
    @param egt_per_bin_df_dict: Mapping between bin and the egt dataframe of it
    @type egt_per_bin_df_dict: dict
    @param threshold_to_egt_dict: mapping between threshold (diff exp) to egt valid in this threshold
    @type threshold_to_egt_dict: dict
    @param batches: list of batches
    @type batches: list
    @param num_bins: number of bins
    @type num_bins: int
    @param min_number_of_gmctypes_per_batch: Minimum number of required gmctype per batch, below that we will filter
    the batch out
    @type min_number_of_gmctypes_per_batch: int
    @param min_number_of_batches_per_gmctypes: minimum number of batches per gmctype, below this we filter the
    gmctype out
    @type min_number_of_batches_per_gmctypes: int
    @param min_expected_umi_threshold: Minimum number of expected umis per gm to make this a valid gm
    @type min_expected_umi_threshold: int
    @param run_mp: should we run multiprocess for each bin, default is False.
    @type run_mp: bool
    @return: Two dicts, the first map the threshold to the glm results and the second map between the threshold and
    the number of batches and egt used in it for calculation
    @rtype: tuple
    """
    kwargs = {
        "egt_per_bin_df_dict": egt_per_bin_df_dict,
        "threshold_to_egt_dict": threshold_to_egt_dict,
        "batches": batches,
        "min_number_of_gmctypes_per_batch": min_number_of_gmctypes_per_batch,
        "min_number_of_batches_per_gmctypes": min_number_of_batches_per_gmctypes,
        "min_expected_umi_threshold": min_expected_umi_threshold,
    }

    if run_mp and min(num_bins, mp.cpu_count() - 2) > 2:
        with mp.Pool(min(num_bins, mp.cpu_count() - 2)) as p:
            calc_results = p.map(
                partial(calculate_alpha_egt_single_bin, **kwargs),
                range(1, num_bins + 1),
            )

    else:
        calc_results = []
        for bin_size in range(1, num_bins + 1):
            calc_results.append(calculate_alpha_egt_single_bin(bin_size, **kwargs))

    # collect the results
    combined_batches_egt_tracker, combined_calculate_results = {}, {
        i: [] for i in range(1, num_bins + 1)
    }
    for bin_size, results, batches_egt_tracker in calc_results:
        combined_batches_egt_tracker[bin_size] = batches_egt_tracker
        combined_calculate_results[bin_size] = results

    return combined_calculate_results, combined_batches_egt_tracker


# +
def get_egt_per_threshold(
    genes_clusters,
    mc_clusters,
    empty_droplet_gcluster_fraction,
    num_bins,
    mat_clean_df,
    mat_clean,
    metacell_df,
    bins_threshold_list,
    thr_max_value=-3,
    thr_value_interval=0.1,
    forbidden_mclusters=[],
    forbidden_gclusters=[],
):
    """
    Collect all the egt information from the mc and cells matrices, and attach it to the thresholds of expression
    @param genes_clusters: mapping of the genes into clusters
    @type genes_clusters: dict{str:list}
    @param mc_clusters: dataframe of type for each mc
    @type mc_clusters: pd.DataFrame
    @param empty_droplet_gcluster_fraction: mapping between the batch and the empty droplets fraction df
    @type empty_droplet_gcluster_fraction: dict
    @param num_bins: number of bins
    @type num_bins: int
    @param mat_clean_df: cells matrix with umis per gene
    @type mat_clean_df: pd.DataFrame
    @param mat_clean: anndata object of the cells
    @type mat_clean: anndata.AnnData
    @param metacell_df: metacell matrix with the amount of umis per gene and mc
    @type metacell_df: pd.DataFrame
    @param bins_threshold_list: List of umi count where each adjust numbers represent bin
    @type bins_threshold_list: list
    @param thr_max_value: Max diff expression to try and add egt to solve the glm
    @type thr_max_value: float
    @param thr_value_interval: intervals of diff expression to add egt to the glm
    @type thr_value_interval: float
    @param forbidden_mclusters: list of mclusters forbidden to use for glm
    @type forbidden_mclusters: list[int]
    @param forbidden_gclusters: list of gclusters forbidden to use for glm
    @type forbidden_gclusters: list[str]
    @return: mapping between bins and the egt information and another mapping between expresison diff threshold and egt
    @rtype: tuple(dict,dict)
    """
    assert all(
        isinstance(mcluster, int) for mcluster in forbidden_mclusters
    ), "forbidden_mclusters must be int"
    assert all(
        isinstance(gcluster, str) for gcluster in forbidden_gclusters
    ), "forbidden_gclusters must be str"

    expressions_diff_df = get_expression_diff_df(
        genes_clusters, mc_clusters, metacell_df
    )

    egt_per_bin_df_dict = {i: {} for i in range(1, num_bins + 1)}
    threshold_to_egt_dict = {}
    calculated_egt = []
    gm_df = get_gm_df_template(mat_clean)
    ndigits = (
        len(str(thr_value_interval).split(".")[1]) if thr_value_interval < 1 else 0
    )

    for current_thr in np.arange(
        np.floor(expressions_diff_df.min().min()),
        thr_max_value + thr_value_interval,
        thr_value_interval,
    ):
        mcluster_list, gcluster_list = np.where(expressions_diff_df <= current_thr)
        for j, mcluster_id in enumerate(mcluster_list):
            gcluster_id = expressions_diff_df.columns[gcluster_list[j]]

            if mcluster_id in forbidden_mclusters or gcluster_id in forbidden_gclusters:
                continue

            label = "Egt_%s_in_%s" % (gcluster_id, mcluster_id)
            if label not in calculated_egt:
                calculated_egt.append(label)
                relevant_genes = genes_clusters[gcluster_id]
                relevent_mc = mc_clusters[mc_clusters.type == mcluster_id].mc.index
                egt_df = get_gcluster_noisy_df(
                    gcluster_id,
                    mcluster_id,
                    relevant_genes,
                    relevent_mc,
                    empty_droplet_gcluster_fraction,
                    mat_clean_df,
                    current_gm_df=gm_df.copy(),
                )
                bin_dict = split_cells_based_on_bin(
                    egt_df, bins_threshold_list=bins_threshold_list
                )

                # todo: make sure works , before it was i in all but we already have i
                for j in bin_dict:
                    egt_per_bin_df_dict[j][label] = bin_dict[j]

        threshold_to_egt_dict[round(current_thr, ndigits)] = calculated_egt.copy()

    return egt_per_bin_df_dict, threshold_to_egt_dict


def get_filtered_eq(glm_equations, valid_batches, valid_gmctype):
    """
    Filter out invalid equations:
        - order the equations columns
        - those without batch information
        - those without egt information
    @param glm_equations: A dataframe with all the equations
    @type glm_equations: pd.Dataframe
    @param valid_batches: a list of all the valid batches
    @type valid_batches: list
    @param valid_gmctype: a list of all the valid gmctype
    @type valid_gmctype: list
    @return: filtered dataframe with all the valid equations to use
    @rtype: pd.DataFrame
    """
    current_equations = glm_equations.loc[
        :, ["obs"] + list(valid_batches) + list(valid_gmctype)
    ]
    current_equations = current_equations[
        np.any(current_equations.loc[:, valid_batches], axis=1)
    ]
    current_equations = current_equations[
        np.any(current_equations.loc[:, valid_gmctype], axis=1)
    ]
    return current_equations


def calculate_alpha_egt_with_shared_egt(
    egt_per_bin_df_dict,
    egt_to_use,
    batches,
    min_expected_umi_threshold=100,
    min_number_of_gmctypes_per_batch=10,
    min_number_of_batches_per_gmctypes=5,
    get_equations=False,
):
    """
    Run glm to find coeff of all the batches and egt while sharing the egt between bins
    @param egt_per_bin_df_dict: Mapping between bin and the egt dataframe of it
    @type egt_per_bin_df_dict: dict
    @param egt_to_use: List of egt to use - they need to be shared across all the bins
    @type egt_to_use; list
    @param batches: list of batches
    @type batches: list
    @param min_number_of_gmctypes_per_batch: Minimum number of required gmctype per batch, below that we will filter
    the batch out
    @type min_number_of_gmctypes_per_batch: int
    @param min_number_of_batches_per_gmctypes: minimum number of batches per gmctype, below this we filter the
    gmctype out
    @type min_number_of_batches_per_gmctypes: int
    @param min_expected_umi_threshold: Minimum number of expected umis per gm to make this a valid gm
    @type min_expected_umi_threshold: int
    @param get_equations: should we also return the equations we used
    @type get_equations: bool
    @return: The glm results with coeff per batch and egt, might also return the equations used in the process
    @rtype: Either pd.DataFrame or tuple(pd.DataFrame,pd.DataFrame)
    """
    assert (
        len(egt_to_use) > min_number_of_gmctypes_per_batch
    ), "egt_to_use < min_number_of_gmctypes_per_batch, can't continue"

    # collect all the egts and rename the batches to represent the bin it was taken from
    bin_size_to_equations = {}
    for bin_size in egt_per_bin_df_dict:
        data_combined = pd.concat(egt_per_bin_df_dict[bin_size]).loc[egt_to_use]
        data_combined = data_combined[
            data_combined.expected >= min_expected_umi_threshold
        ]
        data_combined["batch_name"] = data_combined.index.get_level_values(1)
        glm_equations = build_glm_equation_for_bin(data_combined)

        valid_egt = get_valid_batches_gmctypes(
            data_combined,
            batches,
            min_number_of_gmctypes_per_batch=min_number_of_gmctypes_per_batch,
            min_number_of_batches_per_gmctypes=min_number_of_batches_per_gmctypes,
        )

        assert not valid_egt.empty, (
            "Minimum number of gmctypes or batches is too restricted, no data for batch %s"
            % bin_size
        )

        valid_gmctype = valid_egt.index.get_level_values(0).unique()
        valid_batches = valid_egt.index.get_level_values(1).unique()

        current_equations = get_filtered_eq(glm_equations, valid_batches, valid_gmctype)
        new_columns = []

        for co in current_equations.columns:
            label = "%s_bin_%s" % (co, bin_size) if co in valid_batches else co
            new_columns.append(label)

        current_equations.columns = new_columns
        bin_size_to_equations[bin_size] = current_equations

    # combine all the equations and run the glm
    glm_equations_combined = pd.concat(bin_size_to_equations.values())
    glm_equations_combined = glm_equations_combined.fillna(0)
    glm_results = predict_alphas_and_egt(glm_equations_combined, num_cv=20)
    glm_results["bin"] = [i.split("_")[-1] for i in glm_results.index]
    glm_results.index = [i.split("_bin")[0] for i in glm_results.index]

    if get_equations:
        return glm_results, glm_equations_combined

    return glm_results


# -


def get_empty_droplet_gcluster_fraction(batch_empty_droplets, genes_clusters):
    """
    Get a dataframe representing the fraction of empty droplets per gene cluster in a batch
    @param batch_empty_droplets: Mapping between batch and the empty droplets information
    @type batch_empty_droplets: dict
    @param genes_clusters: Mapping between gene cluster and it's gene
    @type genes_clusters: dict
    @return: dataframe that represent the empty droplets fractions in each gene cluster in the specific batch
    @rtype: pd.DataFrame
    """
    # collect batches_gm information
    empty_droplet_gcluster_fraction = pd.DataFrame(
        index=batch_empty_droplets.keys(),
        columns=genes_clusters.keys(),
        dtype=np.float64,
    )
    for gcluster in genes_clusters:
        gm_genes = genes_clusters[gcluster]
        for batch in batch_empty_droplets:
            batch_empty_droplet_frac = (
                batch_empty_droplets[batch].loc[gm_genes].sum()
                / batch_empty_droplets[batch].sum()
            )
            empty_droplet_gcluster_fraction.loc[
                batch, gcluster
            ] = batch_empty_droplet_frac.values[0]

    return empty_droplet_gcluster_fraction


def plot_predicted_vs_observed(equations, glm_shared_results):
    """
    Plot the difference between the observed umis and the predicted one based on the glm results
    @param equations: All the equations used to predict the coeff
    @type equations:  pd.DataFrame
    @param glm_shared_results: the coefficients prediction from the glm
    @type glm_shared_results: pd.DataFrame
    """
    predicted_mean = np.sum(
        equations[equations.columns[1:]] * glm_shared_results.predicted.values, axis=1
    )
    obs = equations.obs

    init_plt_params(scale=1.5)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(
        111,
        xlabel="Predicted",
        ylabel="Observed",
        title="Observed vs Predicted",
        ylim=(1, max(np.max(obs), np.max(predicted_mean)) * 1.1),
        xlim=(1, max(np.max(obs), np.max(predicted_mean)) * 1.1),
    )

    plt.scatter(x=predicted_mean, y=obs, s=5)
    plt.plot(
        [1, max(np.max(obs), np.max(predicted_mean)) * 1.1],
        [1, max(np.max(obs), np.max(predicted_mean)) * 1.1],
        "k:",
    )

    plt.yscale("log", basey=2)
    plt.xscale("log", basex=2)
    plt.grid()
    plt.show()


def plot_egt_used_per_batch(batches, equations, num_bins, figsize=(10, 15), scale=1.5):
    """
    Plot which egt were used in each batch for every bin
    @param batches: list of batches, will be used to order them
    @type batches: list
    @param equations: Equations used to calculate the coefficients
    @param num_bins: Number of bins
    @type num_bins: int
    @param figsize: the figsize of each heatmap
    @type figsize: tuple(int,int)
    """
    # find all the equations with egt information
    egt_eq = equations.loc[:, [i for i in equations.columns if i.startswith("Egt")]]
    equations.index = egt_eq.columns[np.where(egt_eq != 0)[1]]
    equations.index.name = "gt"

    batches_columns = [i for i in equations.columns if "_bin_" in i]
    only_batches_equations = equations.loc[:, batches_columns]
    summed_egt_obs = only_batches_equations.groupby("gt").sum()
    relevant_ind = summed_egt_obs.apply(lambda row: row[row != 0].index, axis=1)

    for bin_size in range(1, num_bins + 1):
        df = pd.DataFrame(0, index=equations.index.unique(), columns=batches)
        for egt in relevant_ind.index:
            relevant_batches = [
                i.split("_bin_")[0]
                for i in relevant_ind[egt]
                if i.endswith("%s" % bin_size)
            ]
            df.loc[egt, relevant_batches] = 1

        init_plt_params(scale=scale)

        plt.figure(figsize=figsize)
        ax = sb.heatmap(
            df.sort_index(),
            cmap=sb.color_palette("hls", 2),
            linewidths=0.5,
            xticklabels=True,
            yticklabels=True,
        )
        _ = plt.yticks(rotation=0)
        plt.title("Used Egt per batch bin %s" % bin_size)
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(["Unused", "Used"])

        plt.tight_layout()
        plt.show()


def plot_clusters_bins_distribution(
    mat_clean, batches, num_bins, bins_threshold_list, font_scale=2
):
    """
    Plot the distribution of bins size and clusters of cell types in different batches
    @param mat_clean: anndata object of the cells
    @type mat_clean: anndata.AnnData
    @param batches: list of batches, will be sorted by that
    @type batches: list
    @param num_bins: Number of bins
    @type num_bins: int
    @param bins_threshold_list: List of umi count where each adjust numbers represent bin
    @type bins_threshold_list: list
    @param font_scale: scaling the fonts
    @type font_scale: float
    """
    # add bin information to cells
    bins_names = {}
    for i in range(len(bins_threshold_list) - 1):
        bins_names[i + 1] = "(%s,%s]" % (
            int(bins_threshold_list[i]),
            int(bins_threshold_list[i + 1]),
        )
        bins_names["%s" % (i + 1)] = "(%s,%s]" % (
            int(bins_threshold_list[i]),
            int(bins_threshold_list[i + 1]),
        )

    for i in range(len(bins_threshold_list) - 1):
        st = bins_threshold_list[i]
        en = bins_threshold_list[i + 1]
        mat_clean.obs.loc[
            np.logical_and(mat_clean.obs.umi_depth >= st, mat_clean.obs.umi_depth < en),
            "bin_index",
        ] = (
            i + 1
        )

    # plot distribution of cell type annotation (mclusters) per batch
    valid_cells = mat_clean.obs[~np.isnan(mat_clean.obs.bin_index)]
    clustering_count_by_batch = (
        valid_cells.groupby(["batch", "annotation"])
        .agg({"annotation": "count"})
        .unstack(level=1)
    )

    clustering_perc_by_batch = pd.DataFrame(
        data=clustering_count_by_batch.to_numpy(),
        columns=sorted(valid_cells.annotation.unique()),
        index=sorted(valid_cells.batch.unique()),
    )
    clustering_perc_by_batch = clustering_perc_by_batch.div(
        clustering_perc_by_batch.sum(axis=1), axis=0
    )

    init_plt_params(scale=font_scale)
    f = plt.figure(figsize=(16, 8))
    ax = plt.gca()
    clustering_perc_by_batch.loc[batches].plot(
        kind="bar",
        ylim=(0, 1),
        title="% mc clusters per batch",
        ax=ax,
        stacked=True,
        color=sb.color_palette("hls", len(valid_cells.annotation.unique())),
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()

    # plot distribution of bins per batches

    bin_index_count_by_batch = (
        valid_cells.groupby(["batch", "bin_index"])
        .agg({"bin_index": "count"})
        .unstack(level=1)
    )
    bin_index_perc_by_batch = pd.DataFrame(
        data=bin_index_count_by_batch.to_numpy(),
        columns=sorted(valid_cells["bin_index"].unique()),
        index=sorted(valid_cells.batch.unique()),
    )
    bin_index_perc_by_batch = bin_index_perc_by_batch.div(
        bin_index_perc_by_batch.sum(axis=1), axis=0
    )

    f = plt.figure(figsize=(16, 8))
    ax = plt.gca()
    bin_index_perc_by_batch.loc[batches].plot(
        kind="bar", ax=ax, stacked=True, ylim=(0, 1), title="% bin per batch"
    )
    ax.legend(
        ["bin %s: %s" % (i, bins_names[i]) for i in range(1, num_bins + 1)],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    plt.show()


# +
def loglikelihood(obs, expected):
    """
    Calculate log likelihood
    """
    return -expected + np.log(expected) * obs - scs.gammaln(obs)


def plot_ll_of_prediction(
    equations, glm_shared_results, ncols=6, max_alpha=0.25, step_size=0.005
):
    """
    Plot the ll of each prediction of changed in the alpha prediction
    @param equations: A dataframe of all the used equations
    @type equations: pd.DataFrame
    @param glm_shared_results: The results of the final glm calculation
    @type glm_shared_results: pd.DataFrame
    @param ncols: number of columns to plot
    @type ncols: int
    @param max_alpha:  maximum alpha value to test per batch
    @type max_alpha: float
    @param step_size: size of steps for alpha
    @type step_size: float
    """
    assert max_alpha < 1, "max alpha must be < 1"
    # num_cols=12 for pptx

    # gather information about valid batches columns
    batches_for_eq = []
    for non_empty_columns in [
        i for i in equations.apply(lambda row: row[row != 0].index, axis=1)
    ]:
        for col in non_empty_columns:
            if "_bin_" in col:
                batches_for_eq.append(col)

    # calculate the likelihood of each prediction based on different alpha
    likelihood_df = pd.DataFrame(
        index=batches_for_eq, columns=np.arange(step_size, max_alpha, step_size)
    )
    predicted_coeff = glm_shared_results.copy()
    batches_ind = ~predicted_coeff.index.str.startswith("Egt")

    for alpha_value in np.arange(step_size, max_alpha, step_size):
        predicted_coeff.loc[batches_ind, "predicted"] = alpha_value
        predicted_for_alpha = np.sum(
            equations[equations.columns[1:]] * predicted_coeff.predicted.values, axis=1
        )
        likelihood = loglikelihood(equations.obs, predicted_for_alpha)

        # needed for places where the Egt is 0 so we don't get data if we have 0 noise
        likelihood = likelihood.replace(
            -np.inf, np.min(likelihood[likelihood > -np.inf])
        )
        likelihood_df.loc[:, alpha_value] = likelihood.values

    likelihood_df_summed = likelihood_df.groupby(level=0, axis=0).sum()
    likelihood_df_summed = likelihood_df_summed * -1

    # plot the likelihood
    init_plt_params(scale=5)

    num_rows = likelihood_df_summed.shape[0] / ncols
    num_rows = int(num_rows) if int(num_rows) == num_rows else int(num_rows) + 1
    f, axes = plt.subplots(
        num_rows,
        ncols,
        figsize=(12 * ncols, 10 * num_rows),
    )

    axes = axes.reshape(-1)
    plt.tight_layout()
    for i, batch_bin in enumerate(likelihood_df_summed.index):
        ax = axes[i]
        ax.scatter(
            x=likelihood_df_summed.columns * 100,
            y=likelihood_df_summed.loc[batch_bin].values,
            s=200,
        )
        batch, bin_i = batch_bin.split("_bin_")
        bin_results = glm_shared_results[glm_shared_results.bin == bin_i].loc[batch]

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

        if max_alpha > 0.1:
            ax.set_xticks(range(0, int(max_alpha * 100) + 1, 5))
        else:
            ax.set_xticks(range(0, int(max_alpha * 100) + 1, 1))

        start = int(np.log2(max(likelihood_df_summed.loc[batch_bin].values.min(), 1)))
        ax.set_yscale("log", basey=2)
        ax.set_ylim(2**start, 2 ** (start + 8))
        ax.set_yticks([2 ** (start + i) for i in range(0, 8, 2)])
        ax.set_title("%s(bin %s)" % (batch, bin_i))

    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.5, hspace=0.3
    )
    plt.suptitle("- Log-Likelihood on different alphas", fontsize=100)
    plt.show()


# -

# # remove noise

# ## mc pipeline

# +
# Run mc pipeline with batch correction
def get_empty_droplets_dist_per_cell(cell_ad, batch_empty_droplets):
    """
    Generate a dataframe with the empty droplets information per cell and gene
    @param cell_ad: The cells anndata obj
    @type cell_ad: anndata.AnnData
    @param batch_empty_droplets: empty droplets information per batch
    @type batch_empty_droplets: dict
    @return: A dataframe representing the empty droplets information per cell
    """
    batch_probability_array = np.array(
        [batch_empty_droplets[batch] for batch in cell_ad.obs.batch]
    ).squeeze()
    cell_probability_df = pd.DataFrame(
        batch_probability_array,
        columns=cell_ad.var.index,
        index=cell_ad.obs.batch.index,
    )
    return cell_probability_df


def correct_ambient_noise_in_pile_wrapper(empty_droplets_dist_per_cell_df):
    """
    Wrapper function to add the empty droplets information for the mc function which correct the sampled corr matrix
    """

    def correct_ambient_noise_in_pile(adata, fdata, downsampled):
        """
        Correct the ambient noise umi count in the pile by subtracting the umis of each cell by the relevant alpha
        from the batch, the empty droplets information from the batch.
        We make sure not to drop to negative umi count by increasing the number of umis we remove from other genes to
        match the total expected noisy umis
        """
        downsampled_df = pd.DataFrame(
            downsampled, columns=fdata.var.index, index=fdata.obs.index
        )
        multi_factor = 1

        pile_probability_df = empty_droplets_dist_per_cell_df.loc[
            fdata.obs.index, fdata.var.index
        ]
        sampled_prob_df = pile_probability_df.div(
            pile_probability_df.sum(axis=1), axis=0
        )
        noisy_umis = pd.Series(
            (fdata.obs["alpha"] * np.sum(downsampled, axis=1)).values,
            index=fdata.obs.index,
        )

        noisy_matrix = np.multiply(
            multi_factor, np.multiply(noisy_umis[:, np.newaxis], sampled_prob_df)
        )
        dependent_noisy_matrix = np.minimum(noisy_matrix, downsampled_df)
        dependent_noisy_umis = dependent_noisy_matrix.sum(axis=1)
        delta = noisy_umis - dependent_noisy_umis

        valid_cells = np.where(dependent_noisy_umis != 0)  # zero noisy umis
        while np.any(1e-5 < delta.iloc[valid_cells]):
            cells_to_change = delta.iloc[valid_cells][
                1e-5 < delta.iloc[valid_cells]
            ].index
            non_excess_genes = (
                downsampled_df.loc[cells_to_change] > noisy_matrix.loc[cells_to_change]
            )
            excess_umis = (
                noisy_umis.loc[cells_to_change]
                - dependent_noisy_umis.loc[cells_to_change]
            )
            temp_multi_factor = np.maximum(
                1,
                1
                + excess_umis
                / (
                    np.sum(sampled_prob_df[non_excess_genes], axis=1)
                    * np.multiply(noisy_umis, multi_factor)
                ),
            )
            temp_multi_factor[temp_multi_factor.isna()] = 1
            multi_factor *= temp_multi_factor.loc[delta.index]

            noisy_matrix = np.multiply(
                multi_factor[:, np.newaxis],
                np.multiply(noisy_umis[:, np.newaxis], sampled_prob_df),
            )
            dependent_noisy_matrix = np.minimum(noisy_matrix, downsampled_df)
            dependent_noisy_umis = dependent_noisy_matrix.sum(axis=1)
            delta = noisy_umis - dependent_noisy_umis
            valid_cells = np.where(dependent_noisy_umis != 0)  # zero noisy umis

        downsampled -= dependent_noisy_matrix

    return correct_ambient_noise_in_pile


# -

# ## remove noise mc posterior


def get_denoise_metacell_ad(
    mat_ad,
    metacells_ad,
    batches_prediction,
    batch_empty_droplets,
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
    """
    Go over each metacell, bin size and batch and subtract the expected noisy umis based on all the relevant cells.
    Here we make sure not to have negative umis but we won't add those umis to someplace else because we expect the
    aggregation of cells in each metacell to have strong enough statistical power to make this less powerful method
    @param mat_ad: The cell anndata
    @type mat_ad: ad.AnnData
    @param metacells_ad: The mc anndata we want to annotate
    @type metacells_ad: ad.AnnData
    @param batches_prediction: The dataframe with prediction of all the bins
    @type batches_prediction: pd.DataFrame
    @param batch_empty_droplets: empty droplets information per batch
    @type batch_empty_droplets: dict
    @param valid_obs: valid obs columns to move from the old to the new anndata
    @param valid_var: valid vars column to move from the old to the new anndata
    @param blacklist_obs: blacklist obs we shouldn't move
    @param blacklist_var: blacklist vars we shouldn't move
    @return:A new anndata file after removin the noise
    @type: anndata.Anndata
    """
    metacell_df = mc.ut.get_vo_frame(metacells_ad).copy()

    agg_cells_info = mat_ad.obs.groupby(["batch", "bin_index", "metacell"]).agg(
        {"umi_depth_corrected": "sum"}
    )
    for batch_name in batches_prediction.index.unique():
        batch_empty_droplets_frac = (
            batch_empty_droplets[batch_name].loc[metacells_ad.var.index]
            / batch_empty_droplets[batch_name].sum()
        )
        for bin_i in mat_ad.obs[mat_ad.obs.batch == batch_name].bin_index.unique():
            batch_df = batches_prediction[batches_prediction.index == batch_name]
            selected_bin = bin_i
            while selected_bin not in batch_df.bin.values:
                if selected_bin > max(batch_df.bin.values):
                    selected_bin = max(batch_df.bin.values)
                else:
                    selected_bin += 1

            rel_batch = batch_df[batch_df.bin == selected_bin]
            alpha = rel_batch.predicted.values[0] / 100
            batch_bin_mc_umis = agg_cells_info.loc[batch_name, bin_i] * alpha
            noise_per_mc = (
                batch_bin_mc_umis.umi_depth_corrected.values.reshape(-1, 1)
                * batch_empty_droplets_frac.T.values
            )
            metacell_df.iloc[batch_bin_mc_umis.index] -= noise_per_mc

    # make sure we don't have zeros in our matrix: max(0, obs-noise)
    metacell_df[metacell_df < 0] = 0

    given_vs_valid_obs = list(
        set(metacells_ad.obs.columns) - set(valid_obs) - set(blacklist_obs)
    )
    given_vs_valid_var = list(
        set(metacells_ad.var.columns) - set(valid_var) - set(blacklist_var)
    )
    if len(given_vs_valid_obs):
        print(
            "Found non valid obs and didn't passed them, insert manualy to pass:\n%s"
            % (" ".join(given_vs_valid_obs))
        )

    if len(given_vs_valid_var):
        print(
            "Found non valid var and didn't passed them, insert manualy to pass:\n%s"
            % (" ".join(given_vs_valid_var))
        )

    valid_obs = metacells_ad.obs.columns & valid_obs
    valid_var = metacells_ad.var.columns & valid_var
    denoise_mc_ad = ad.AnnData(
        X=metacell_df.to_numpy(),
        obs=metacells_ad.obs[valid_obs],
        var=metacells_ad.var[valid_var],
    )
    return denoise_mc_ad


def add_batch_bin_corrected_umis_information(mat_ad, batches_prediction):
    """
    Add information layers for each cell for noisy correction
    @param mat_ad: The cell anndata we want to annotate
    @type mat_ad: ad.AnnData
    @param batches_prediction: The dataframe with prediction of all the bins
    @type batches_prediction: pd.DataFrame
    @return: The cell anndata we want to add batch information
    @rtype: anndata.Anndata
    """
    # adding the corrected umi depth so we'll know how to calculate the noise
    mc_median = mat_ad.obs.groupby("metacell").agg({"umi_depth": np.median})
    mat_ad.obs["umi_depth_corrected"] = np.min(
        [
            mat_ad.obs.umi_depth.values.reshape(-1, 1),
            mc_median.loc[mat_ad.obs.metacell] * 2,
        ],
        axis=0,
    )

    for bin_size in batches_prediction.bin.unique():
        st = batches_prediction.loc[
            batches_prediction.bin == bin_size, "min_umis_depth"
        ].values[0]
        en = batches_prediction.loc[
            batches_prediction.bin == bin_size, "max_umis_depth"
        ].values[0]
        mat_ad.obs.loc[
            np.logical_and(mat_ad.obs.umi_depth >= st, mat_ad.obs.umi_depth < en),
            "bin_index",
        ] = bin_size

    # adding the too large or small cells to the closest bin to calculate the noise
    mat_ad.obs.loc[
        mat_ad.obs.umi_depth <= batches_prediction.min_umis_depth.min(), "bin_index"
    ] = 1
    mat_ad.obs.loc[
        mat_ad.obs.umi_depth >= batches_prediction.min_umis_depth.max(), "bin_index"
    ] = batches_prediction.bin.max()

    for batch_name in mat_ad.obs.batch.unique():
        if batch_name not in batches_prediction.index.unique():
            mat_ad.obs.loc[mat_ad.obs.batch == batch_name, "alpha"] = 0
            continue

        for bin_index in mat_ad.obs[mat_ad.obs.batch == batch_name].bin_index.unique():
            rel_bin = bin_index
            rel_batch = batches_prediction.loc[batch_name]

            # only one option - take the closet we have
            if len(rel_batch.shape) == 1:
                rel_bin = rel_batch.bin

            else:
                # take the closest one, prefer to take larger then smaller
                diff = np.abs(rel_bin - rel_batch.bin.unique())
                rel_bin = np.max(rel_batch.bin.unique()[diff == np.min(diff)])

            alpha = batches_prediction[
                np.logical_and(
                    batches_prediction.index == batch_name,
                    batches_prediction.bin == rel_bin,
                )
            ].predicted[0]
            mat_ad.obs.loc[
                np.logical_and(
                    mat_ad.obs.batch == batch_name, mat_ad.obs.bin_index == bin_index
                ),
                "alpha",
            ] = (
                alpha / 100
            )

    return mat_ad


def get_corr_on_fg(metacell_df, mat_clean):
    """
    Calculate corr on the feature genes
    @param mat_clean: anndata object of the cells
    @type mat_clean: anndata.AnnData
    @param metacell_df: metacell matrix with the amount of umis per gene and mc
    @type metacell_df: pd.DataFrame
    @return: The correlation matrix
    @rtype: pd.Dataframe
    """
    genes = metacell_df.columns & mat_clean.var[mat_clean.var.feature_gene != 0].index
    metacell_df_fractions = convert_mc_df_to_logged_frac(metacell_df)
    metacell_df_fractions_normalized = metacell_df_fractions.subtract(
        metacell_df_fractions.median(), axis=1
    )
    metacell_df_fractions_normalized_fg = metacell_df_fractions_normalized.loc[:, genes]

    corr = metacell_df_fractions_normalized_fg.T.corr()
    corr.index = corr.index.astype(int)
    corr.columns = corr.columns.astype(int)

    return corr


# +
def get_batchy_mc_with_annotation_df(
    metacell_batch_composition, annotation_df, batchy_thr=0.5, cell_types_colors=None
):
    """
    Get all the batchy metacell (based on a filter) with annotation of the batch. A metacell is consider batchy
    if the fraction of one batch is larger then `batchy_thr`
    @param metacell_batch_composition: Information about each mc and the batch composition
    @type metacell_batch_composition: pd.DataFrame
    @param annotation_df: the annotation dataframe
    @type annotation_df: pd.DataFrame
    @param batchy_thr: The threshold to consider a metacell batchy
    @type batchy_thr: float
    @param cell_types_colors: Dataframe representing the color for each cell type
    @type cell_types_colors: pd.DataFrame
    @return: A dataframe with all the batchy mc, the batch which they are batch and color annotation based on
    batch and cell type
    @rtype: pd.DataFrame
    """
    annotation_df = annotation_df.copy()
    if isinstance(cell_types_colors, pd.DataFrame):
        annotation_df["color"] = annotation_df.apply(
            lambda row: cell_types_colors.loc[row.cell_type, "color"], axis=1
        )

    else:
        mc_clusters_rgb = sb.color_palette("hls", len(annotation_df.cell_type.unique()))
        mc_clusters_rgb_dict = {
            t: mc_clusters_rgb[i]
            for i, t in enumerate(sorted(annotation_df.cell_type.unique()))
        }
        annotation_df["color"] = annotation_df.apply(
            lambda row: mc_clusters_rgb_dict[row.cell_type], axis=1
        )

    annotation_df.loc[pd.isna(annotation_df.cell_type), "cell_type"] = "unknown"
    batchy_mc = metacell_batch_composition.umi_depth_corrected[
        metacell_batch_composition.umi_depth_corrected.values > batchy_thr
    ].index

    batchy_mc_df = pd.DataFrame(
        index=[i[0] for i in batchy_mc if i[0] in annotation_df.index]
    )
    batchy_mc_df["batch"] = [i[1] for i in batchy_mc if i[0] in annotation_df.index]
    batchy_mc_df["cell_type"] = batchy_mc_df.apply(
        lambda row: annotation_df.loc[row.index, "cell_type"]
    )

    annotation_df.loc[:, "batch"] = "mix"
    annotation_df.loc[batchy_mc_df.index, "batch"] = batchy_mc_df.batch.values
    batches_rgb = sb.color_palette("hls", len(annotation_df.batch.unique()))
    batches_colors_rgs_dict = {
        t: batches_rgb[i] for i, t in enumerate(sorted(annotation_df.batch.unique()))
    }
    batches_colors_rgs_dict["mix"] = "white"

    annotation_df["batch_color"] = annotation_df.apply(
        lambda row: batches_colors_rgs_dict[row.batch], axis=1
    )

    def f(row):
        a = annotation_df.loc[annotation_df.batch == row.batch, "batch_color"].unique()
        if len(a):
            return a[0]
        return "white"

    batchy_mc_df["batch_color"] = batchy_mc_df.apply(lambda row: f(row), axis=1)

    def f1(row):
        i = annotation_df.loc[
            annotation_df.cell_type == row.cell_type, "color"
        ].unique()
        v = i[0] if i.shape[0] > 0 else "white"
        return v

    batchy_mc_df["color"] = batchy_mc_df.apply(lambda row: f1(row), axis=1)
    return batchy_mc_df, annotation_df


def get_mc_order_by_type_batch(mc_df, corr_matrix):
    """
    Sort the metacell based on type and then batch
    @param mc_df: the metacell dataframe with umi information
    @type mc_df: pd.DataFrame
    @param corr_matrix: the correlation matrix to sort
    @type corr_matrix: pd.Dataframe
    @return: two lists, one give the groups order (first by type and then by batch) and the second is the batches order
    @rtype: list, list
    """
    groups_order = []
    batches_order = []
    for ctype in sorted(mc_df.cell_type.unique()):
        for batch in sorted(mc_df[mc_df.cell_type == ctype].batch.unique()):
            group_index = mc_df[
                np.logical_and(mc_df.cell_type == ctype, mc_df.batch == batch)
            ].index
            if len(group_index) == 0:
                continue
            elif len(group_index) == 1:
                groups_order.append(group_index[0])
                batches_order.append(batch)
            else:
                group_linkage = hierarchy.linkage(
                    corr_matrix.loc[group_index, group_index], method="centroid"
                )
                group_linkage_order = leaves_list(group_linkage)
                groups_order.extend(group_index[group_linkage_order])
                batches_order.extend([batch] * len(group_index))

    return groups_order, batches_order


def get_mc_order_by_batch_type(mc_df, corr_matrix):
    """
    Sort the metacell based on batch and then by cell type
    @param mc_df: the metacell dataframe with umi information
    @type mc_df: pd.DataFrame
    @param corr_matrix: the correlation matrix to sort
    @type corr_matrix: pd.Dataframe
    @return: two lists, one give the groups order (first by batch and then by type) and the second is the batches order
    @rtype: list, list
    """
    groups_order = []
    batches_order = []
    for batch in sorted(mc_df.batch.unique()):
        for ctype in sorted(mc_df.cell_type.unique()):
            group_index = mc_df[
                np.logical_and(mc_df.cell_type == ctype, mc_df.batch == batch)
            ].index
            if len(group_index) == 0:
                continue
            elif len(group_index) == 1:
                groups_order.append(group_index[0])
                batches_order.append(batch)
            else:
                group_linkage = hierarchy.linkage(
                    corr_matrix.loc[group_index, group_index], method="centroid"
                )
                group_linkage_order = leaves_list(group_linkage)
                groups_order.extend(group_index[group_linkage_order])
                batches_order.extend([batch] * len(group_index))

    return groups_order, batches_order


# +
def plot_corr_clustermap(
    corr_matrix,
    groups_order,
    color_df,
    title,
    batches_order,
    vmin=-1,
    vmax=1,
    figsize=(75, 75),
    nrows=4,
    scale=1,
):
    """
    Plot the correlation clustermap ordered by a specific order with two color map - the cell type and the batch
    @param corr_matrix: The correlation between each mc
    @type corr_matrix: pd.DataFrame
    @param groups_order: list of metacell index to order by
    @type groups_order: list
    @param color_df: A dataframe with all the coloring information, both about the cell type colors and the batch color
    @type color_df: pd.DataFrame
    @param title: The title for the plot
    @type title: str
    @param batches_order: The order of the batches to plot
    @type batches_order: list
    @param vmin: min value for the clutstermap legend
    @type vmin: float
    @param vmax: max value for the clustermap legend
    @type vmax: float
    @param figsize: figure size for the clustermap
    @type figsize: tuple(int, int)
    @param nrows: number of rows for the legends
    @type nrows: int
    """
    batches_colors = [
        color_df[color_df.batch == i]["batch_color"].unique()[0]
        for i in batches_order
        if i in color_df.batch.unique()
    ]

    init_plt_params(is_clustermap=True, scale=scale)
    g = sb.clustermap(
        corr_matrix.iloc[groups_order, groups_order],
        col_colors=[color_df.loc[groups_order]["color"], batches_colors],
        row_colors=[color_df.loc[groups_order]["color"], batches_colors],
        cmap="RdBu_r",
        linewidths=0,
        col_cluster=False,
        row_cluster=False,
        row_linkage=groups_order,
        col_linkage=groups_order,
        vmin=vmin,
        vmax=vmax,
        cbar_pos=(0.1, 0.1, 0.01, 0.3),
        center=0,
        figsize=figsize,
        yticklabels=False,
        xticklabels=False,
    )

    g.fig.suptitle(title)

    for label in sorted(color_df.batch.unique()):
        g.ax_col_dendrogram.bar(
            0,
            0,
            color=color_df[color_df.batch == label].batch_color.unique(),
            label=label,
            linewidth=0,
        )
    g.ax_col_dendrogram.legend(
        loc="upper left",
        ncol=int(len(color_df.batch.unique()) / nrows) + 1,
        title="batch",
        bbox_to_anchor=(0, 0.8),
    )

    for label in sorted(color_df.cell_type.unique()):
        g.ax_row_dendrogram.bar(
            0,
            0,
            color=color_df[color_df.cell_type == label].color.unique(),
            label=label,
            linewidth=0,
        )

    g.ax_row_dendrogram.legend(loc="upper right", ncol=2, title="cell types")
    plt.show()
    return g


def plot_noisy_denoise_umap(noisy_mc_ad, denoise_mc_ad, color_df):
    """
    Plot two umaps, first for the noisy metacells and then  for the denoise one
    """
    mc.pl.compute_umap_by_features(noisy_mc_ad, min_dist=0.5, random_seed=234)
    plot_annotation_on_umap(noisy_mc_ad, color_df)

    mc.pl.compute_umap_by_features(denoise_mc_ad, min_dist=0.5, random_seed=234)
    plot_annotation_on_umap(denoise_mc_ad, color_df)


def plot_top_changes(
    metacell_df,
    denoise_mc_df,
    annotation_df,
    color_df,
    num_genes=50,
    fg=[],
    legend_ncols=3,
):
    """
    Plot heatmap with the log delta of changes between the metacell expression and the denoise one
    @param metacell_df: the metacell dataframe
    @type metacell_df: pd.DataFrame
    @param denoise_mc_df: the denoise metacell dataframe
    @type denoise_mc_df: pd.DataFrame
    @param annotation_df: annotation dataframe of cell types
    @type annotation_df: pd.DataFrame
    @param color_df: color dataframe with cell type to color
    @type color_df: pd.DataFrame
    @param num_genes: number of genes to plot
    @type num_genes: int
    @param fg: list of feature genes, will be marked with *
    @type fg: list[str]
    """
    logged_delta = convert_mc_df_to_logged_frac(
        metacell_df
    ) - convert_mc_df_to_logged_frac(denoise_mc_df)
    top_changed_genes = (
        logged_delta.max().sort_values(ascending=False).index[:num_genes]
    )
    df = logged_delta[sorted(top_changed_genes)].iloc[
        annotation_df.sort_values("cell_type").index
    ]

    if len(fg):
        col = []
        for c in sorted(top_changed_genes):
            if c in fg:
                col.append("%s*" % c)
            else:
                col.append(c)

        df.columns = col

    init_plt_params(is_clustermap=True)
    g = sb.clustermap(
        df.T,
        col_colors=[
            color_df.iloc[annotation_df.sort_values("cell_type").index].color.values
        ],
        method="average",
        col_cluster=False,
        row_cluster=True,
        cmap="YlGnBu",
        figsize=(300, 200),
        linewidths=0,
        cbar_pos=(0.1, 0.2, 0.03, 0.5),
    )

    for label in sorted(color_df.cell_type.unique()):
        g.ax_row_dendrogram.bar(
            0,
            0,
            color=color_df[color_df.cell_type == label].color.unique(),
            label=label,
            linewidth=0,
        )

    g.ax_row_dendrogram.legend(loc="upper right", ncol=legend_ncols, title="cell types")
    g.fig.suptitle("log(frac_noisy)) - log(frac_denoise) of top changed genes")
    plt.show()
