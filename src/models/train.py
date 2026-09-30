# ============================================================
# WINE MODEL
#
# File:
#     src/models/train.py
#
# PURPOSE:
#     Contains our machine-learning model classes.
#
# OOP CONCEPTS:
#     🔵 Abstraction
#     🟢 Inheritance
# ============================================================


# ------------------------------------------------------------
# Python tools for creating abstract classes.
# ------------------------------------------------------------

from abc import ABC, abstractmethod


# ------------------------------------------------------------
# Random Forest machine-learning algorithm.
# ------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 🔵 ABSTRACTION
#
# BaseWineModel defines the common requirements for
# every wine model.
# ============================================================

class BaseWineModel(ABC):

    # --------------------------------------------------------
    # Every model must implement train().
    # --------------------------------------------------------

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    # --------------------------------------------------------
    # Every model must implement predict().
    # --------------------------------------------------------

    @abstractmethod
    def predict(self, X_test):
        pass


# ============================================================
# 🟢 INHERITANCE
#
# RandomForestWineModel inherits from BaseWineModel.
# ============================================================

class RandomForestWineModel(BaseWineModel):

    # --------------------------------------------------------
    # Constructor
    # --------------------------------------------------------

    def __init__(self, n_estimators=100):

        # Store the number of trees.
        self.n_estimators = n_estimators

        # The actual model will be created during training.
        self.model = None


    # --------------------------------------------------------
    # Train the model.
    # --------------------------------------------------------

    def train(self, X_train, y_train):

        # Create the Random Forest model.
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=42
        )

        # Train the model.
        self.model.fit(
            X_train,
            y_train
        )


    # --------------------------------------------------------
    # Make predictions.
    # --------------------------------------------------------

    def predict(self, X_test):

        # Use the trained model to make predictions.
        return self.model.predict(X_test)