import json
import logging
import random

import numpy as np
import pandas as pd
from quantile_regression import QuantileRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scikeras.wrappers import KerasRegressor
from tensorflow import keras
from tensorflow.keras import layers



class DigitalTwin:
    """Digital Twin class that learns behaviour of your assets.
    The :class:`DigitalTwin` leverages :class:`QuantileRegression`
    class to learn machine power consumption.
    Parameters
    ----------
    configurations : str
        The different setups the machine assets can have, e.g.
        looking at compressors it could be ["C1", "C2", "C1C2"]
    quantile : float, default=0.5
        The quantile that the model tries to predict. It must be strictly
        between 0 and 1. If 0.5 (default), the model predicts the 50%
        quantile, i.e. the median.
    Attributes
    ----------
    configurations : array of shape (n_configurations,)
        The different setups the machine assets can have
    quantile : float, default=0.5
        The quantile that the model tries to predict. It must be strictly
        between 0 and 1. If 0.5 (default), the model predicts the 50%
        quantile, i.e. the median.
    models : array of shape (n_configurations,)
        Trained model per configuration.
    test_mae_errors : array of shape (n_configurations,)
        Test MAE per configuration's trained model.
    train_mae_errors : array of shape (n_features,)
        Train MAE per configuration's trained model.
    Examples
    --------
    >>> twin = DigitalTwin(
    >>>     configurations=df["configuration_column"].unique().tolist()
    >>> ).train(df=df, features=["x", "y"], target="z", config_col="configuration_column")
    """

    def __init__(
        self,
        configurations: list,
        features: list,
        target: str,
        config_col: str,
        quantile: float = 0.5,
    ):
        """Initialize the class.

        Args:
            :configurations:    the different asset configurations
            :quantile:          the quantile of the quantile forrest, default: 0.5

        """
        self.configurations = configurations
        self.quantile = quantile
        self.models = {}
        self.test_mae_errors = {}
        self.train_mae_errors = {}
        self.features = features
        self.target = target
        self.config_col = config_col
        
    def _quantile_regression(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        degree: int = 2,
        verbose: bool = True,
        remove_outliers: bool = False,
    ) -> QuantileRegression:
        """Train the digital twin based on all possible configurations.
        Parameters
        ----------
        df           : {array-like, DataFrame} of shape (n_samples, n_features+n_target+n_configuration)
                       Training DataFrame having features, target and configuration.
        features     : {array-like} string array with the feature column names
        target       : {string} the target column
        config_col   : {string} configuration column name in DataFrame
        test_size    : {float} ratio of train-test-split
        random_state : {float} a random state for reproducability
        degree       : {float} polynomial degree you want to add as additional features
        verbose      : {bool} whether to print training evaluation or not
        Returns
        -------
        self : object
            Returns self.
        """
        for config in self.configurations:
            temp = df[df[self.config_col] == config]
            X = temp[self.features]
            y = temp[self.target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            estimators = [
                ("scale", RobustScaler()),
                ("poly", PolynomialFeatures(degree=degree)),
                ("regressor", QuantileRegression(quantile=self.quantile)),
            ]

            pipeline = Pipeline(estimators)

            model = pipeline.fit(X_train, y_train)

            self.models[config] = model

            test_pred = model.predict(X_test)
            train_pred = model.predict(X_train)

            self.test_mae_errors[config] = mean_absolute_error(test_pred, y_test)
            self.train_mae_errors[config] = mean_absolute_error(train_pred, y_train)

            if verbose:
                print(config)
                print("------------")
                print("In Sample Errors")
                print(f"Combination {config} yields: Mean = {np.mean(y_train)}")
                print(f"Combination {config} yields: Stddev = {np.std(y_train)}")
                print(
                    f"Combination {config} yields: RMSE = {np.sqrt(mean_squared_error(train_pred, y_train))}"
                )
                print(
                    f"Combination {config} yields: MAE = {self.train_mae_errors[config]}"
                )

                print("Out of Sample Errors")
                print(f"Combination {config} yields: Mean = {np.mean(y_test)}")
                print(f"Combination {config} yields: Stddev = {np.std(y_test)}")
                print(
                    f"Combination {config} yields: RMSE = {np.sqrt(mean_squared_error(test_pred, y_test))}"
                )
                print(
                    f"Combination {config} yields: MAE = {self.test_mae_errors[config]}"
                )
                print("------------")

            model = pipeline.fit(X, y)
            self.models[config] = model

        return self
    
    def _baseline_model(self):
        """Baseline Keras model
        Parameters
        ----------
        Returns
        -------
        model : keras.Sequential()
            Returns the model.
        """
        # create model
        model = keras.Sequential()
        model.add(layers.Dense(256, input_dim=26, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dense(32, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dense(64, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dense(16, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dense(8, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model
    
    def _keras_regression(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        degree: int = 2,
        optimizer: str="adam",
        optimizer__learning_rate: float=0.001,
        epochs: int=1000,
        verbose: int=0):
        
        regression = KerasRegressor(
            model=self._baseline_model(),
            optimizer=optimizer,
            optimizer__learning_rate=optimizer__learning_rate,
            epochs=epochs,
            verbose=verbose,
        )

        X = df[self.features+[self.config_col]].reset_index(drop=True)
        y = df[self.target].reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=X[self.config_col])

        numeric_features = self.features
        numeric_steps = [
            ("scale", MinMaxScaler()),
            ("poly", PolynomialFeatures(degree=degree)),
        ]
        numeric_transformer = Pipeline(steps=numeric_steps)

        categorical_features = [self.config_col]
        categorical_steps = [
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]
        categorical_transformer = Pipeline(steps=categorical_steps)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        self.keras_reg = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ("regressor", regression),
            ])

        self.keras_reg = self.keras_reg.fit(X_train, y_train)

        test_pred = self.keras_reg.predict(X_test)
        train_pred = self.keras_reg.predict(X_train)

        test_mae_errors = mean_absolute_error(test_pred, y_test)
        train_mae_errors = mean_absolute_error(train_pred, y_train)
        
        for config in self.configurations:
            test_index = X_test.config == config
            temp = X_test[test_index]
            test_pred = self.keras_reg.predict(temp)
            oos = mean_absolute_error(test_pred, y_test[test_index])

            train_index = X_train.config == config
            temp = X_train[train_index]
            train_pred = self.keras_reg.predict(temp)
            ins = mean_absolute_error(train_pred, y_train[train_index])

            self.test_mae_errors[config] = oos
            self.train_mae_errors[config] = ins
            if verbose:
                    print(config)
                    print("------------")
                    print("In Sample Errors")
                    print(f"Combination {config} yields: Mean = {np.mean(y_train[train_index])}")
                    print(f"Combination {config} yields: Stddev = {np.std(y_train[train_index])}")
                    print(
                        f"Combination {config} yields: RMSE = {np.sqrt(mean_squared_error(train_pred, y_train[train_index]))}"
                    )
                    print(
                        f"Combination {config} yields: MAE = {self.train_mae_errors[config]}"
                    )

                    print("Out of Sample Errors")
                    print(f"Combination {config} yields: Mean = {np.mean(y_test)}")
                    print(f"Combination {config} yields: Stddev = {np.std(y_test)}")
                    print(
                        f"Combination {config} yields: RMSE = {np.sqrt(mean_squared_error(test_pred, y_test[test_index]))}"
                    )
                    print(
                        f"Combination {config} yields: MAE = {self.test_mae_errors[config]}"
                    )
                    print("------------")
                
        return self

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        degree: int = 2,
        verbose: bool = True,
        optimizer: str="adam",
        optimizer__learning_rate: float=0.001,
        epochs: int=1000,
        algorithm: str="keras_regression",
        remove_outliers: bool = False,
    ) -> QuantileRegression:
        """Train the digital twin based on all possible configurations.
        Parameters
        ----------
        df           : {array-like, DataFrame} of shape (n_samples, n_features+n_target+n_configuration)
                       Training DataFrame having features, target and configuration.
        features     : {array-like} string array with the feature column names
        target       : {string} the target column
        config_col   : {string} configuration column name in DataFrame
        test_size    : {float} ratio of train-test-split
        random_state : {float} a random state for reproducability
        degree       : {float} polynomial degree you want to add as additional features
        verbose      : {bool} whether to print training evaluation or not
        Returns
        -------
        self : object
            Returns self.
        """
        if remove_outliers:
            df = self.remove_outliers(df=df)
        if algorithm == "quantile_regression":
            return self._quantile_regression(
                df=df,
                test_size=test_size,
                random_state=random_state,
                degree=degree,
                verbose=verbose,)
        else:
            return self._keras_regression(
                df=df,
                test_size=test_size,
                random_state=random_state,
                degree=degree,
                optimizer=optimizer,
                optimizer__learning_rate=optimizer__learning_rate,
                epochs=epochs,
                verbose=verbose,)

        return self

    def remove_outliers(
        self,
        df: pd.DataFrame,
        n_estimators: int = 200,
        max_samples: float = 0.75,
        contamination: float = 0.01,
        max_features: float = 0.75,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Remove outliers of DataFrame
        Parameters
        ----------
        df : {array-like, DataFrame} of shape (n_samples, n_features+n_target)
            Training DataFrame having features, target and configuration.
        n_estimators : int, default=100
            The number of base estimators in the ensemble.
        max_samples : "auto", int or float, default="auto"
            The number of samples to draw from X to train each base estimator.
                - If int, then draw `max_samples` samples.
                - If float, then draw `max_samples * X.shape[0]` samples.
                - If "auto", then `max_samples=min(256, n_samples)`.
            If max_samples is larger than the number of samples provided,
            all samples will be used for all trees (no sampling).
        contamination : 'auto' or float, default='auto'
            The amount of contamination of the data set, i.e. the proportion
            of outliers in the data set. Used when fitting to define the threshold
            on the scores of the samples.
                - If 'auto', the threshold is determined as in the
                  original paper.
                - If float, the contamination should be in the range (0, 0.5].
            .. versionchanged:: 0.22
               The default value of ``contamination`` changed from 0.1
               to ``'auto'``.
        max_features : int or float, default=1.0
            The number of features to draw from X to train each base estimator.
                - If int, then draw `max_features` features.
                - If float, then draw `max_features * X.shape[1]` features.
        n_jobs : int, default=None
            The number of jobs to run in parallel for both :meth:`fit` and
            :meth:`predict`. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors. See :term:`Glossary <n_jobs>` for more details.
        random_state : int, RandomState instance or None, default=None
            Controls the pseudo-randomness of the selection of the feature
            and split values for each branching step and each tree in the forest.
            Pass an int for reproducible results across multiple function calls.
            See :term:`Glossary <random_state>`.
        Returns
        -------
        self : pandas.core.series.Series
            Returns pandas series with historic power consumption.
        """
        X = df[self.features + [self.target]].reset_index(drop=True)
        self.iso = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        ).fit(X)
        df["contamination"] = self.iso.predict(df[self.features + [self.target]])
        df = df[df["contamination"] == 1].reset_index(drop=True)
        if "dttm" in df.columns:
            df.index = df.dttm
        return df

    def predict_efficiency(self, df: pd.DataFrame, algorithm: str="quantile_regression") -> pd.Series:
        """Predict historical asset efficiency
        Parameters
        ----------
        df           : {array-like, DataFrame} of shape (n_samples, n_features+n_target+n_configuration)
                       Training DataFrame having features, target and configuration.
        Returns
        -------
        self : pandas.core.series.Series
            Returns pandas series with historic power consumption.
        """
        if algorithm == "quantile_regression":
            temp = df.copy("deep")
            temp["model_prediction"] = 0
            uconfig = df[self.config_col].unique()
            for config in uconfig:
                tmp = temp[temp[self.config_col] == config]
                pred = self.models[config].predict(tmp[self.features])
                temp.loc[temp[self.config_col] == config, "model_prediction"] = pred
            return temp["model_prediction"]
        else:
            return self.keras_reg.predict(df)

    def create_staging_table(self, load, staging_rules):
        """Create staging table.
        Parameters
        ----------
        load               : {array-like, Series} target load column
        staging_rules      : {array-like} string array needs to be of size (2, )
        Returns
        -------
        self : list
            Returns power consumption.
        """
        staging = np.zeros_like(load)
        for switch_point in staging_rules[1]:
            staging += np.where(load > switch_point, 1, 0)

        return [staging_rules[0][s] for s in staging.astype("int")]

    def simulate_conditions(
        self, load_conditions, staging_rules, features, target, config_col="config"
    ):
        """Simulate load conditions.
        Parameters
        ----------
        load_conditions    : {array-like, DataFrame} the last x weeks load conditions
        staging_rules      : {array-like} string array needs to be of size (2, )
        features           : {array-like} string array with the feature column names
        target             : {string} the target column
        config_col         : {string} configuration column name in DataFrame
        Returns
        -------
        self : float
            Returns power consumption.
        """
        load_conditions[config_col] = self.create_staging_table(
            load=load_conditions[target], staging_rules=staging_rules
        )

        grouped = load_conditions.groupby(config_col)

        power_consumption = 0

        for name, group in grouped:

            power_predictions = (
                self.models[name].predict(group[features]) + self.test_mae_errors[name]
            )

            power_consumption += np.nansum(power_predictions)

        return power_consumption

    def range_subset(self, range1, range2):
        """Whether range1 is in range2
        Parameters
        ----------
        range1: {range-like, range} of shape (n_samples,)
        range2: {range-like, range} of shape (n_samples,)
                       Training DataFrame having features, target and configuration.
        Returns
        -------
        One of True or False : bool
            Returns whether range1 is in range2 or not
        """
        if not range1:
            return True  # empty range is subset of anything
        if not range2:
            return False  # non-empty range can't be subset of empty range
        if len(range1) > 1 and range1.step % range2.step:
            return False  # must have a single value or integer multiple step
        return range1.start in range2 and range1[-1] in range2

    def objective_function_quantile(
        self, clf, flow, press, n=100, boundary={}, train_error_weight=1.0
    ):
        """Function to be minimize -> total energy consumption per compressor setting given shopfloor constraints
        Parameters
        ----------
        clf: {list} of shape (n_compressor_settings,)
        flow: {list} of shape (2,)
        press: {list} of shape (2,)
        n: {integer} number of random entries to be generated
        boundary: {dict} contstraints per compressor setting
        train_error_weight: {float} factor to add/subtract/eliminate training error, e.g.
            1.0: add MAE error from test set
            0.0: ignore error
            -1.0: subtract MAE error from test set
        Returns
        -------
        power_consumption : dict
            Returns summed power consumption per compressor setting
        """
        power_consumption = {}

        # Generate random entries in range
        flow_vec = np.random.randint(low=flow[0], high=flow[1], size=n)
        press_vec = np.random.randint(low=press[0], high=press[1], size=n)

        # For each selected setting option
        for name in clf:
            # If compressor setting has boundary check and either skip or ignore
            if name in boundary:
                bounds = boundary[name]
                if not self.range_subset(
                    range1=range(flow[0], flow[1]), range2=range(bounds[0], bounds[1])
                ):
                    continue
            power_predictions = 0
            # Predict and sum overall power consumption
            power_predictions = self.models[name].predict(
                pd.DataFrame(
                    data=np.transpose([flow_vec, press_vec]), columns=self.features
                )
            ) + (train_error_weight * self.test_mae_errors[name])
            power_consumption[name] = np.nansum(power_predictions)
                
        return power_consumption
    
    def objective_function_keras(
        self, clf, flow, press, n=100, boundary={}, train_error_weight=1.0
    ):
        """Function to be minimize -> total energy consumption per compressor setting given shopfloor constraints
        Parameters
        ----------
        clf: {list} of shape (n_compressor_settings,)
        flow: {list} of shape (2,)
        press: {list} of shape (2,)
        n: {integer} number of random entries to be generated
        boundary: {dict} contstraints per compressor setting
        train_error_weight: {float} factor to add/subtract/eliminate training error, e.g.
            1.0: add MAE error from test set
            0.0: ignore error
            -1.0: subtract MAE error from test set
        Returns
        -------
        power_consumption : dict
            Returns summed power consumption per compressor setting
        """
        power_consumption = {}

        # Generate random entries in range
        flow_vec = np.random.randint(low=flow[0], high=flow[1], size=n)
        press_vec = np.random.randint(low=press[0], high=press[1], size=n)

        # For each selected setting option
        for name in clf:
            # If compressor setting has boundary check and either skip or ignore
            if name in boundary:
                bounds = boundary[name]
                if not self.range_subset(
                    range1=range(flow[0], flow[1]), range2=range(bounds[0], bounds[1])
                ):
                    continue
            power_predictions = 0
            # Predict and sum overall power consumption
            confs = np.repeat(a=name, repeats=n)
            D = self.keras_reg["preprocessor"].transform(
                pd.DataFrame(
                    data=np.transpose([flow_vec, press_vec, confs]), columns=self.features+[self.config_col]
                )
            )
            power_predictions = self.keras_reg["regressor"].model(D)
            power_consumption[name] = np.nansum(power_predictions)
                
        return power_consumption

    def make_trial(
        self,
        clf,
        flow,
        press,
        n=100,
        no_of_trials=1000,
        boundary={},
        train_error_weight=1.0,
        algorithm="quantile_regression",
    ):
        """Function to be minimize -> total energy consumption per compressor setting given shopfloor constraints
        Parameters
        ----------
        clf: {list} of shape (n_compressor_settings,)
        flow: {list} of shape (2,)
        press: {list} of shape (2,)
        n: {integer} number of random entries to be generated
        no_of_trials: {integer} number of trials to run
        boundary: {dict} contstraints per compressor setting
        train_error_weight: {float} factor to add/subtract/eliminate training error, e.g.
            1.0: add MAE error from test set
            0.0: ignore error
            -1.0: subtract MAE error from test set
        Returns
        -------
        power_consumption : dict
            Returns summed power consumption per compressor setting
        """
        trials = []
        for t in range(no_of_trials):
            energy = 0
            if algorithm == "quantile_regression":
                energy = self.objective_function_quantile(
                    clf=clf,
                    flow=flow,
                    press=press,
                    n=n,
                    boundary=boundary,
                    train_error_weight=train_error_weight,
                )
            else:
                energy = self.objective_function_keras(
                    clf=clf,
                    flow=flow,
                    press=press,
                    n=n,
                    boundary=boundary,
                    train_error_weight=train_error_weight,
                )
                
            # Setting is not compliant
            if len(energy) == 0:
                return {}
            key = min(energy, key=energy.get)
            trials.append({key: energy[key]})
        return trials

    def run_monte_carlo(
        self,
        clf,
        flow,
        press,
        n=100,
        no_of_trials=1000,
        boundary={},
        train_error_weight=1.0,
        algorithm="quantile_regression",
    ):
        """Run Monte Carlo simulation based on constraints
        Parameters
        ----------
        clf: {list} of shape (n_compressor_settings,)
        flow: {list} of shape (2,)
        press: {list} of shape (2,)
        n: {integer} number of random entries to be generated
        no_of_trials: {integer} number of trials to run
        boundary: {dict} contstraints per compressor setting
        train_error_weight: {float} factor to add/subtract/eliminate training error, e.g.
            1.0: add MAE error from test set
            0.0: ignore error
            -1.0: subtract MAE error from test set
        Returns
        -------
        power_consumption : dict
            Returns summed power consumption per compressor setting
        """
        results = []

        while len(clf) > 0:
            counts = {}
            minima = {}
            trials = self.make_trial(
                clf=clf,
                flow=flow,
                press=press,
                n=n,
                no_of_trials=no_of_trials,
                boundary=boundary,
                train_error_weight=train_error_weight,
                algorithm=algorithm,
            )
            if len(trials) > 0:
                for t in trials:
                    key = min(t, key=t.get)
                    if key not in counts:
                        counts[key] = 1
                        minima[key] = round(t[key], 0)
                    else:
                        counts[key] += 1
                        minima[key] = round(min(minima[key], t[key]), 0)
                asset = max(counts, key=counts.get)
                results.append({asset: minima[asset]})
                clf.remove(asset)
            else:
                clf = []

        return results
