"""
RGlmSolver

Object that perform an estimation of coefficents using generalized linear model (GLM) using the zetadiv libary.
This allow us to use poisson loss estimation with identity link function while still forcing non-negative coefficents.
Sadly there is no such python version yet so we have to use the R version to do this.
"""


import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri

from MCNoise import ambient_logger


numpy2ri.activate()
pandas2ri.activate()


class RGlmSolver(object):
    def __init__(self) -> None:
        """
        Initialized and upload an R environment and the solver
        """
        self.logger = ambient_logger.logger()

        self.r_zetadiv = importr("zetadiv")
        self.r_base = importr("base")

    def fit_coefficents_based_on_equations(
        self,
        equations: pd.DataFrame,
        starting_estimation_for_batches: float = 0.02,
        starting_estimation_for_pgm: float = 1e-7,
    ) -> pd.Series:
        """
        Use the observed values in the equations and fit the best coefficients, given the other data in the equations.
        Currently, a GLM with non-negative constraints, identity link function, and Poisson loss is used.

        :param equations: Holds all the data needed for fitting; the first column is observed and considered the 'y' part of the equations.
        :type equations: pd.DataFrame

        :param starting_estimation_for_batches: Starting value for coefficients for the noise levels, defaults to 0.02.
        :type starting_estimation_for_batches: float, optional

        :param starting_estimation_for_pgm: Starting value for coefficients of native expression, defaults to 1e-7.
        :type starting_estimation_for_pgm: float, optional

        :return: Either nan series if failed to fit or the best fit for those equations.
        :rtype: pd.Series
        """
        estimations_results = pd.Series(np.nan, index=equations.columns[1:])

        # Sometimes after using subset of the dataset in cv we will have a column which can't be solved.
        valid_columns = self.get_valid_column_for_estimation(equations)

        x, y = equations.loc[:, valid_columns], equations["observed"]
        estimation_start_values_vector = self._get_estimation_start_values_vector(
            x.columns,
            starting_estimation_for_batches=starting_estimation_for_batches,
            starting_estimation_for_pgm=starting_estimation_for_pgm,
        )

        try:
            model = self.r_zetadiv.glm_fit_cons(
                x=x,
                y=y,
                cons=1,
                intercept=False,
                family=robjects.r.poisson(link="identity"),
                start=estimation_start_values_vector,
            )
            model_results = dict(zip(model.names, list(model)))

        except rpy2.rinterface_lib.embedded.RRuntimeError as ex:
            self.logger.warning("Unable to solve the given set of equations, provided exception %s" %ex)
            return estimations_results

        estimations_results.loc[x.columns] = model_results["coefficients"]
        return estimations_results

    def get_valid_column_for_estimation(self, equations: pd.DataFrame) -> list:
        """
        Sadly the constraint version of zetadiv does not handle columns without enough data to estimate and raise an error.
        To prevent this, we run the constraint version of another function, which will yield a false estimation but will tell us which columns are valid for calculation by the correct version.
        We will then use this to check which columns are valid for later use.

        :param equations:
            The set of equations we want to solve with the solver.
            The first column must be named "observed" and the rest of the columns represent the data matching the coefficients we want to find.
            In our specific case, we will have mostly 0 in all columns per row except two columns - one for the noise levels and the other for the native expression.
        :type equations: pd.DataFrame

        :return: The name of all the columns we will be able to calculate their coefficents.
        :rtype: list
        """
        # Reduce 1 because of the first column which represent y.
        number_of_coefficents_columns = equations.shape[1] - 1
        pre_equations_columns = equations.columns

        model_r_results = self.r_zetadiv.glm_cons(
            "observed ~ . - 1",
            data=equations,
            cons=1,
            family=robjects.r.poisson(link="identity"),
            method="lm.fit",
        )

        model_results = dict(zip(model_r_results.names, list(model_r_results)))
        valid_columns = list(
            equations.columns[1:][~np.isnan(model_results["coefficients"])]
        )

        # If several columns were dropped, we need to make sure we can still solve for all other columns.
        # Do the validation process again.
        if len(valid_columns) != number_of_coefficents_columns:
            valid_columns = self.get_valid_column_for_estimation(
                equations.loc[:, ["observed"] + valid_columns]
            )
            self.logger.info("Not enough information to solve for: %s, will try to solve for the rest" %", ".join(list(set(pre_equations_columns) - set(["observed"] + valid_columns))))

        return valid_columns

    def _get_estimation_start_values_vector(
        self,
        coefficents_name: pd.Index,
        starting_estimation_for_batches: float,
        starting_estimation_for_pgm: float,
    ) -> np.ndarray:
        """
        Generate a vector with the starting values for different column estimations.
        This can be divided into two parts:
        1. The batches estimation is governed by the starting_estimation_for_batches parameter. Its length is based on all columns without the word Pgm.
        2. The native expression estimation is governed by the starting_estimation_for_pgm parameter. Its length is all the columns with the word Pgm.


        :param coefficents_name: This is the names of the columns of the equations.
        :type coefficents_name: pd.Index

        :param starting_estimation_for_batches: The starting value of the coefficients  representing noise levels of batches.
        :type starting_estimation_for_batches: float

        :param starting_estimation_for_pgm: The starting value of the coefficients representing the native expression of the cells-genes cluster.
        :type starting_estimation_for_pgm: float

        :return: array with the length of all the parameters needed to estimate the starting value for each index.
        :rtype: np.ndarray
        """
        starting_values_vector = np.empty(len(coefficents_name))
        starting_values_vector.fill(starting_estimation_for_pgm)
        starting_values_vector[
            ~coefficents_name.str.startswith("Pgm")
        ] = starting_estimation_for_batches
        return starting_values_vector
