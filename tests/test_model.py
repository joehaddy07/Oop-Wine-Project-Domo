#BaseWineModel
     #│
     #├── RandomForestWineModel
     #│
     #└── DecisionTreeWineModel

#Now we test the OOP model classes.

# OOP concepts are being applied:

#🔵 Abstraction
#🟢 Inheritance
#🔴 Polymorphism

# ============================================================
# TEST MACHINE LEARNING MODELS
#
# File:
#     tests/test_model.py
#
# PURPOSE:
#     Test our Random Forest and Decision Tree model classes.
#
#
# OOP CONCEPTS BEING TESTED:
#
#     🔵 ABSTRACTION
#     🟢 INHERITANCE
#     🔴 POLYMORPHISM
# ============================================================


# ============================================================
# IMPORT DATA LOADER
# ============================================================

from src.data.load_data import WineDataLoader


# ============================================================
# IMPORT PREPROCESSOR
# ============================================================

from src.features.preprocessing import WinePreprocessor


# ============================================================
# IMPORT RANDOM FOREST
# ============================================================

from src.models.train import RandomForestWineModel


# ============================================================
# IMPORT DECISION TREE
# ============================================================

from src.models.predict import DecisionTreeWineModel


# ============================================================
# HELPER FUNCTION
#
# This prevents us from repeating the data preparation code
# in every test.
# ============================================================

def prepare_data():

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
    # Create features.
    # --------------------------------------------------------

    X = preprocessor.create_features(
        wine_dataset
    )


    # --------------------------------------------------------
    # Create target.
    # --------------------------------------------------------

    y = preprocessor.create_target(
        wine_dataset
    )


    # --------------------------------------------------------
    # Split dataset.
    # --------------------------------------------------------

    return preprocessor.split_data(
        X,
        y
    )


# ============================================================
# TEST 1
#
# TEST RANDOM FOREST
# ============================================================

def test_random_forest_model():

    # --------------------------------------------------------
    # Prepare data.
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        prepare_data()
    )


    # --------------------------------------------------------
    # Create Random Forest object.
    # --------------------------------------------------------

    model = RandomForestWineModel(
        n_estimators=10
    )


    # --------------------------------------------------------
    # Train model.
    #
    # 🔴 POLYMORPHISM
    #
    # We call train() through the model interface.
    # --------------------------------------------------------

    model.train(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Make predictions.
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Verify that predictions were produced.
    # --------------------------------------------------------

    assert len(predictions) == len(
        y_test
    )


# ============================================================
# TEST 2
#
# TEST DECISION TREE
# ============================================================

def test_decision_tree_model():

    # --------------------------------------------------------
    # Prepare data.
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        prepare_data()
    )


    # --------------------------------------------------------
    # Create Decision Tree object.
    # --------------------------------------------------------

    model = DecisionTreeWineModel(
        max_depth=5
    )


    # --------------------------------------------------------
    # Train the model.
    #
    # 🔴 POLYMORPHISM
    #
    # Decision Tree has its own implementation of train().
    # --------------------------------------------------------

    model.train(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Make predictions.
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Verify prediction count.
    # --------------------------------------------------------

    assert len(predictions) == len(
        y_test
    )


# ============================================================
# TEST 3
#
# TEST POLYMORPHISM
# ============================================================
#
# We can use different model objects through the same
# interface:
#
#     train()
#     predict()
#
# ============================================================

def test_model_polymorphism():

    # --------------------------------------------------------
    # Prepare data.
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        prepare_data()
    )


    # --------------------------------------------------------
    # Put both model objects into the same list.
    #
    # They are different classes.
    #
    # But they provide the same interface.
    # --------------------------------------------------------

    models = [

        RandomForestWineModel(
            n_estimators=10
        ),

        DecisionTreeWineModel(
            max_depth=5
        )
    ]


    # --------------------------------------------------------
    # Loop through both models.
    # --------------------------------------------------------

    for model in models:

        # ----------------------------------------------------
        # Same train() method.
        #
        # 🔴 POLYMORPHISM
        # ----------------------------------------------------

        model.train(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # Same predict() method.
        #
        # 🔴 POLYMORPHISM
        # ----------------------------------------------------

        predictions = model.predict(
            X_test
        )


        # ----------------------------------------------------
        # Verify predictions.
        # ----------------------------------------------------

        assert len(predictions) == len(
            y_test
        )
