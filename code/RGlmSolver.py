"""
RGlmSolver

Object that perform an estimation of coefficents using generalized linear model (GLM) using the zetadiv libary.
This allow us to use poisson loss estimation with identity link function while still forcing non-negative coefficents.
Sadly there is no such python version yet so we have to use the R version to do this.
"""


import logging

import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects as robjects
from rpy2.rinterface import RRuntimeWarning
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()


class RGlmSolver(object):
    def __init__(self) -> None:
        """
        Initialized and upload an R environment and the solver
        """
        self.r_zetadiv = importr("zetadiv")
        self.r_base = importr("base")

    def fit_coefficents_based_on_equations(
        self,
        equations: pd.DataFrame,
        starting_estimation_for_batches=0.02,
        starting_estimation_for_pgm=1e-7,
    ) -> pd.Series:
        """Use the observed values in the equations and fit the best coefficents to it given the other data in the equations.
        This is being done using a GLM with non-negative constraints and identity link function and poisson loss.

        Args:
            equations (pd.DataFrame): Holds all the data needed for fitting, the first column is called observed and is consider to be the 'y' part of the equations.
            starting_estimation_for_batches (float, optional): Starting value for coefficents for the noise levels. Defaults to 0.02.
            starting_estimation_for_pgm (_type_, optional): Strting value for coefficents of native expression. Defaults to 1e-7.

        Returns:
            pd.Series: Either nan series if failed to fit or the best fit for those equations.
        """

        estimations_results = pd.Series(np.nan, index=equations.columns[1:])

        x, y = equations.loc[:, equations.columns[1:]], equations["observed"]
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
            # TODO : logging
            print(ex)
            return estimations_results

        estimations_results.loc[x.columns] = model_results["coefficients"]
        return estimations_results

    def get_valid_column_for_estimation(self, equations: pd.DataFrame) -> list:
        """Sadly the constraint version of zetadiv doesn't handle columns without enough data to estimte and just crash.
        To prevent this we run the constraint version of another function, which will yield false estimation but will tell us which columns are valid for calculation by the true version.
        We will then use this to check which columns are valid for later use.

        Args:
            equations (pd.DataFrame): The set of equations we want to solve with the solver.
            The first column must be names "observed" and the rest of the columns represent the data matching the coefficents we want to find.
            In our specific case we will have mostly 0 in all columns per row except two columns - one for the noise levels and the other for the native expression.

        Returns:
            list: All the columns which we might be able to calculate their coefficents in the constraint version.
        """
        number_of_coefficents_columns = (
            equations.shape[1] - 1
        )  # Minus 1 because of observed column

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

        return valid_columns

    def _get_estimation_start_values_vector(
        self,
        coefficents_name: pd.Index,
        starting_estimation_for_batches: float,
        starting_estimation_for_pgm: float,
    ) -> np.ndarray:
        """Generate a vector with the starting values for different columns estimation.
        This can be divided to two parts:
        1. The batches estimation, which is being govern by the starting_estimation_for_batches parameter. It's length is based on all columns without the word Pgm
        2. The native expression estimation, which is being govern by the starting_estimation_for_pgm parameter. It's length is all the columns with the word pgm.

        Args:
            coefficents_name (pd.Index): This is the names of the columns of the equations.
            starting_estimation_for_batches (float): The starting value of the coefficents representing noise levels of batches.
            starting_estimation_for_pgm (float): The starting vlaue of the coefficents representing native expression of cells-genes cluster pair.

        Returns:
            np.ndarray: Vector with the length of all the paramaters needed to estimate with the starting value for estiamtion in each index.
        """
        starting_values_vector = np.empty(len(coefficents_name))
        starting_values_vector.fill(starting_estimation_for_pgm)
        starting_values_vector[
            ~coefficents_name.str.startswith("Pgm")
        ] = starting_estimation_for_batches
        return starting_values_vector
