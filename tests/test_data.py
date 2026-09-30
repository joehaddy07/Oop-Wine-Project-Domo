
#We want to verify the following:
#=================================
# The CSV can be loaded.
# The dataset isn't empty
# The expected quality column exists.
# Features are created correctly.
# The target is created correctly.
# The train/test split works.


# ============================================================
# TEST DATA COMPONENTS
#
# File:
#     tests/test_data.py
#
# PURPOSE:
#     Test our data loading and preprocessing classes.
#
# ============================================================


# ============================================================
# IMPORT PANDAS
# ============================================================

import pandas as pd


# ============================================================
# IMPORT OUR DATA LOADER
# ============================================================

from src.data.load_data import WineDataLoader


# ============================================================
# IMPORT OUR PREPROCESSOR
# ============================================================

from src.features.preprocessing import WinePreprocessor


# ============================================================
# TEST 1
#
# TEST DATA LOADING
# ============================================================

def test_load_data():

    # --------------------------------------------------------
    # Create our data loader.
    # --------------------------------------------------------

    loader = WineDataLoader(
        "data/raw/wine.csv"
    )


    # --------------------------------------------------------
    # Load the wine dataset.
    # --------------------------------------------------------

    wine_dataset = loader.load_data()


    # --------------------------------------------------------
    # Make sure the result is a pandas DataFrame.
    #
    # This verifies that our loader returned the expected
    # type of object.
    # --------------------------------------------------------

    assert isinstance(
        wine_dataset,
        pd.DataFrame
    )


    # --------------------------------------------------------
    # Make sure the dataset contains rows.
    #
    # An empty dataset would mean something went wrong.
    # --------------------------------------------------------

    assert len(wine_dataset) > 0


# ============================================================
# TEST 2
#
# TEST TARGET COLUMN
# ============================================================

def test_quality_column_exists():

    # --------------------------------------------------------
    # Create the loader.
    # --------------------------------------------------------

    loader = WineDataLoader(
        "data/raw/wine.csv"
    )


    # --------------------------------------------------------
    # Load the dataset.
    # --------------------------------------------------------

    wine_dataset = loader.load_data()


    # --------------------------------------------------------
    # Verify that our target column exists.
    # --------------------------------------------------------

    assert "quality" in wine_dataset.columns


# ============================================================
# TEST 3
#
# TEST FEATURE CREATION
# ============================================================

def test_create_features():

    # --------------------------------------------------------
    # Load the dataset.
    # --------------------------------------------------------

    loader = WineDataLoader(
        "data/raw/wine.csv"
    )

    wine_dataset = loader.load_data()


    # --------------------------------------------------------
    # Create the preprocessor object.
    # --------------------------------------------------------

    preprocessor = WinePreprocessor()


    # --------------------------------------------------------
    # Create X.
    # --------------------------------------------------------

    X = preprocessor.create_features(
        wine_dataset
    )


    # --------------------------------------------------------
    # The "quality" column should NOT be inside X.
    #
    # Why?
    #
    # Because "quality" is our target.
    # --------------------------------------------------------

    assert "quality" not in X.columns


    # --------------------------------------------------------
    # Make sure X contains data.
    # --------------------------------------------------------

    assert len(X) > 0


# ============================================================
# TEST 4
#
# TEST TARGET CREATION
# ============================================================

def test_create_target():

    # --------------------------------------------------------
    # Load dataset.
    # --------------------------------------------------------

    loader = WineDataLoader(
        "data/raw/wine.csv"
    )

    wine_dataset = loader.load_data()


    # --------------------------------------------------------
    # Create preprocessor.
    # --------------------------------------------------------

    preprocessor = WinePreprocessor()


    # --------------------------------------------------------
    # Create target.
    # --------------------------------------------------------

    y = preprocessor.create_target(
        wine_dataset
    )


    # --------------------------------------------------------
    # Make sure the target has the same number of rows
    # as the original dataset.
    # --------------------------------------------------------

    assert len(y) == len(
        wine_dataset
    )


    # --------------------------------------------------------
    # Our target should contain only:
    #
    #     0 = Bad Quality
    #     1 = Good Quality
    # --------------------------------------------------------

    assert set(
        y.unique()
    ).issubset(
        {0, 1}
    )


# ============================================================
# TEST 5
#
# TEST TRAIN/TEST SPLIT
# ============================================================

def test_split_data():

    # --------------------------------------------------------
    # Load dataset.
    # --------------------------------------------------------

    loader = WineDataLoader(
        "data/raw/wine.csv"
    )

    wine_dataset = loader.load_data()


    # --------------------------------------------------------
    # Create preprocessor.
    # --------------------------------------------------------

    preprocessor = WinePreprocessor()


    # --------------------------------------------------------
    # Create features and target.
    # --------------------------------------------------------

    X = preprocessor.create_features(
        wine_dataset
    )

    y = preprocessor.create_target(
        wine_dataset
    )


    # --------------------------------------------------------
    # Split the data.
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        preprocessor.split_data(
            X,
            y
        )
    )


    # --------------------------------------------------------
    # Verify that the training data contains rows.
    # --------------------------------------------------------

    assert len(X_train) > 0


    # --------------------------------------------------------
    # Verify that the testing data contains rows.
    # --------------------------------------------------------

    assert len(X_test) > 0


    # --------------------------------------------------------
    # Verify that X and y were split consistently.
    # --------------------------------------------------------

    assert len(X_train) == len(y_train)

    assert len(X_test) == len(y_test)
