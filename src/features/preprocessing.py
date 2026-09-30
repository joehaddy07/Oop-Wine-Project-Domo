# ============================================================
#preprocessor
#     │
#     ├── test_size = 0.20
#     │
#     ├── random_state = 2
#     │
#     ├── create_features()
#     │
#     ├── create_target()
#     │
#     └── split_data()
#
#===============================================================
# WINE DATA PREPROCESSOR
#
# File:
#     src/features/preprocessing.py
#
# PURPOSE:
#     This class prepares the wine dataset for machine
#     learning.
#
# RESPONSIBILITIES:
#     1. Create X (features)
#     2. Create y (target)
#     3. Split data into training and testing sets
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CLASS: WinePreprocessor
# ============================================================
# A class is a blueprint for creating objects.
#
# This class creates objects that know how to preprocess
# our Wine Quality dataset.
# ============================================================

class WinePreprocessor:

    # ========================================================
    # __init__()
    # ========================================================
    # This special method runs automatically when we create
    # a WinePreprocessor object.
    #
    # We don't need any information from the user yet, so
    # there are no additional parameters.
    # ========================================================

    def __init__(self, test_size: float = 0.2, random_state: int = 42):

        # ----------------------------------------------------
        # Store the test size inside the object.
        #
        # 0.20 means 20% of the data will be used for testing.
        # ----------------------------------------------------

        self.test_size = test_size

        # ----------------------------------------------------
        # Store the random state inside the object.
        #
        # random_state makes our train/test split reproducible.
        #
        # If we run the code again with the same value, we get
        # the same split.
        # ----------------------------------------------------

        self.random_state = random_state


    # ========================================================
    # METHOD: create_features()
    # ========================================================
    # This method creates X.
    #
    # X contains the input features that the ML model will
    # use to make a prediction.
    # ========================================================

    def create_features(self, wine_dataset):

        # ----------------------------------------------------
        # Remove the "quality" column.
        #
        # "quality" is our target.
        #
        # axis=1 means:
        #     Remove a COLUMN.
        # ----------------------------------------------------

        X = wine_dataset.drop(
            "quality",
            axis=1
        )

        # ----------------------------------------------------
        # Return the features.
        # ----------------------------------------------------

        return X


    # ========================================================
    # METHOD: create_target()
    # ========================================================
    # This method creates y.
    #
    # Our original dataset contains quality scores such as:
    #
    # 3, 4, 5, 6, 7, 8
    #
    # We convert them into two classes:
    #
    # quality >= 7  → 1 → Good Wine
    #
    # quality < 7   → 0 → Bad Wine
    # ========================================================

    def create_target(self, wine_dataset):

        # ----------------------------------------------------
        # Get the "quality" column.
        # ----------------------------------------------------

        y = wine_dataset["quality"].apply(
            lambda quality: 1 if quality >= 7 else 0
        )

        # ----------------------------------------------------
        # Return the target values.
        # ----------------------------------------------------

        return y


    # ========================================================
    # METHOD: split_data()
    # ========================================================
    # This method divides our dataset into:
    #
    # X_train
    # X_test
    # y_train
    # y_test
    #
    # Training data:
    #     Used to teach the model.
    #
    # Testing data:
    #     Used to evaluate the model on unseen data.
    # ========================================================

    def split_data(self, X, y):

        # ----------------------------------------------------
        # train_test_split() comes from scikit-learn.
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(

            # ------------------------------------------------
            # Input features
            # ------------------------------------------------

            X,

            # ------------------------------------------------
            # Target
            # ------------------------------------------------

            y,

            # ------------------------------------------------
            # 20% goes to the testing dataset.
            # ------------------------------------------------

            test_size=self.test_size,

            # ------------------------------------------------
            # Makes the split reproducible.
            # ------------------------------------------------

            random_state=self.random_state
        )

        # ----------------------------------------------------
        # Return all four datasets.
        # ----------------------------------------------------

        return X_train, X_test, y_train, y_test