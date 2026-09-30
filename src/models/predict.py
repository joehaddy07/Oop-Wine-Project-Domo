#============================================================
# Define the base model class
#============================================================

from abc import ABC, abstractmethod

class BaseWineModel(ABC):
    """
    OOP TOPIC: ABSTRACTION + INHERITANCE
    This abstract class defines WHAT every model must do:
        - train()
        - predict()
    Concrete models (RandomForest, XGBoost, etc.) will INHERIT from this class.
    """
#============================================================
# TRAINING AND PREDICTION
# OOP TOPIC: ABSTRACTION
# This class defines the interface for all wine models.
# Python gives us this functionality so that we can create a blueprint for other classes.
#============================================================
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

# ============================================================
# 🔴 POLYMORPHISM
#
# DecisionTreeWineModel is another type of wine model.
#
# It also inherits from BaseWineModel.
#
# Therefore, it must provide:
#
#     train()
#     predict()
#
# However, the implementation is different from
# RandomForestWineModel.
# ============================================================
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeWineModel(BaseWineModel):

    # --------------------------------------------------------
    # Constructor
    # --------------------------------------------------------

    def __init__(self, max_depth=None):

        # Store the maximum depth of the tree.
        self.max_depth = max_depth

        # The actual model will be created during training.
        self.model = None


    # --------------------------------------------------------
    # Train the Decision Tree.
    # --------------------------------------------------------

    def train(self, X_train, y_train):

        # Create the Decision Tree model.
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=42
        )

        # Train the model using the training data.
        self.model.fit(
            X_train,
            y_train
        )


    # --------------------------------------------------------
    # Make predictions using the Decision Tree.
    # --------------------------------------------------------

    def predict(self, X_test):

        # Use the trained Decision Tree
        # to make predictions.
        return self.model.predict(X_test)