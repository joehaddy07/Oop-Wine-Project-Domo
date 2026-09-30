
# ============================================================
# WINE MACHINE LEARNING PIPELINE
#
# File:
#     src/pipeline.py
#
# PURPOSE:
#     Coordinate the complete Wine Quality ML workflow.
#
#
# OOP CONCEPTS:
#
#     🔵 ABSTRACTION
#         The pipeline works with models through:
#
#             train()
#             predict()
#
#         The pipeline does not need to know the internal
#         implementation of each algorithm.
#
#
#     🟢 INHERITANCE
#         RandomForestWineModel and DecisionTreeWineModel
#         inherit from BaseWineModel.
#
#
#     🔴 POLYMORPHISM
#         Both models provide:
#
#             train()
#             predict()
#
#         but each model can implement those methods
#         differently.
#
#
#     🟡 COMPOSITION
#         WinePipeline HAS-A:
#
#             WineDataLoader
#             WinePreprocessor
#             RandomForestWineModel
#             DecisionTreeWineModel
#             ModelEvaluator
#             MLflowTracker
#
#
#     🟣 ENCAPSULATION
#         Each class hides its internal implementation.
#         WinePipeline delegates responsibilities to those
#         classes.
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
# IMPORT RANDOM FOREST MODEL
# ============================================================

from src.models.train import RandomForestWineModel


# ============================================================
# IMPORT DECISION TREE MODEL
# ============================================================

from src.models.predict import DecisionTreeWineModel


# ============================================================
# IMPORT MODEL EVALUATOR
# ============================================================

from src.evaluation.evaluate import ModelEvaluator


# ============================================================
# IMPORT MLFLOW TRACKER
# ============================================================

from src.mlflow.tracking import MLflowTracker


# ============================================================
# WINE PIPELINE
# ============================================================

class WinePipeline:

    """
    Coordinates the complete Wine Quality ML workflow.

    🟡 COMPOSITION:

    WinePipeline HAS-A:

        WineDataLoader
        WinePreprocessor
        RandomForestWineModel
        DecisionTreeWineModel
        ModelEvaluator
        MLflowTracker
    """


    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        data_path
    ):

        # ====================================================
        # 🟡 COMPOSITION
        #
        # WinePipeline HAS-A WineDataLoader.
        # ====================================================

        self.data_loader = WineDataLoader(
            data_path
        )


        # ====================================================
        # 🟡 COMPOSITION
        #
        # WinePipeline HAS-A WinePreprocessor.
        # ====================================================

        self.preprocessor = WinePreprocessor()


        # ====================================================
        # 🟡 COMPOSITION
        #
        # WinePipeline HAS-A RandomForestWineModel.
        #
        # 🟢 INHERITANCE
        #
        # RandomForestWineModel inherits from BaseWineModel.
        # ====================================================

        self.random_forest_model = RandomForestWineModel(
            n_estimators=100
        )


        # ====================================================
        # 🟡 COMPOSITION
        #
        # WinePipeline HAS-A DecisionTreeWineModel.
        #
        # 🟢 INHERITANCE
        #
        # DecisionTreeWineModel inherits from BaseWineModel.
        #
        # 🔴 POLYMORPHISM
        #
        # This model also provides:
        #
        #     train()
        #     predict()
        #
        # just like RandomForestWineModel.
        # ====================================================

        self.decision_tree_model = DecisionTreeWineModel(
            max_depth=5
        )


        # ====================================================
        # 🟡 COMPOSITION
        #
        # WinePipeline HAS-A ModelEvaluator.
        # ====================================================

        self.evaluator = ModelEvaluator(
            output_dir="./artifacts"
        )


        # ====================================================
        # 🟡 COMPOSITION
        #
        # WinePipeline HAS-A MLflowTracker.
        # ====================================================

        self.mlflow_tracker = MLflowTracker(
            tracking_uri="http://localhost:5000",
            experiment_name="Wine_Quality_Predictions"
        )


    # ========================================================
    # RUN PIPELINE
    # ========================================================

    def run(self):

        # ====================================================
        # STEP 1
        #
        # LOAD DATA
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 1: LOAD DATA"
        )

        print(
            "================================================"
        )


        # ----------------------------------------------------
        # Ask WineDataLoader to load the dataset.
        #
        # 🟣 ENCAPSULATION
        #
        # We don't directly use pd.read_csv() here.
        # ----------------------------------------------------

        wine_dataset = self.data_loader.load_data()


        # ====================================================
        # STEP 2
        #
        # CREATE FEATURES
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 2: CREATE FEATURES"
        )

        print(
            "================================================"
        )


        X = self.preprocessor.create_features(
            wine_dataset
        )


        # ====================================================
        # STEP 3
        #
        # CREATE TARGET
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 3: CREATE TARGET"
        )

        print(
            "================================================"
        )


        y = self.preprocessor.create_target(
            wine_dataset
        )


        # ====================================================
        # STEP 4
        #
        # SPLIT DATA
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 4: SPLIT DATA"
        )

        print(
            "================================================"
        )


        X_train, X_test, y_train, y_test = (
            self.preprocessor.split_data(
                X,
                y
            )
        )


        # ====================================================
        # STEP 5
        #
        # START MLFLOW RUN
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 5: START MLFLOW RUN"
        )

        print(
            "================================================"
        )


        self.mlflow_tracker.start_run(
            run_name="OOP_Wine_Quality_Models"
        )


        # ====================================================
        # STEP 6
        #
        # TRAIN RANDOM FOREST
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 6: TRAIN RANDOM FOREST"
        )

        print(
            "================================================"
        )


        # ----------------------------------------------------
        # 🔴 POLYMORPHISM
        #
        # We call:
        #
        #     train()
        #
        # on the Random Forest object.
        #
        # ----------------------------------------------------

        self.random_forest_model.train(
            X_train,
            y_train
        )


        # ====================================================
        # STEP 7
        #
        # RANDOM FOREST PREDICTIONS
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 7: RANDOM FOREST PREDICTIONS"
        )

        print(
            "================================================"
        )


        # ----------------------------------------------------
        # 🔴 POLYMORPHISM
        #
        # RandomForestWineModel provides predict().
        # ----------------------------------------------------

        rf_predictions = self.random_forest_model.predict(
            X_test
        )


        # ====================================================
        # STEP 8
        #
        # EVALUATE RANDOM FOREST
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 8: RANDOM FOREST EVALUATION"
        )

        print(
            "================================================"
        )


        rf_results = self.evaluator.evaluate(
            y_test,
            rf_predictions
        )


        # ----------------------------------------------------
        # Display Random Forest accuracy.
        # ----------------------------------------------------

        print(
            "\nRandom Forest Accuracy:"
        )

        print(
            f"{rf_results['accuracy']:.4f}"
        )


        # ====================================================
        # STEP 9
        #
        # TRAIN DECISION TREE
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 9: TRAIN DECISION TREE"
        )

        print(
            "================================================"
        )


        # ----------------------------------------------------
        # 🔴 POLYMORPHISM
        #
        # Notice that we use exactly the same method:
        #
        #     train()
        #
        # But this time the object is a Decision Tree.
        #
        # ----------------------------------------------------

        self.decision_tree_model.train(
            X_train,
            y_train
        )


        # ====================================================
        # STEP 10
        #
        # DECISION TREE PREDICTIONS
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 10: DECISION TREE PREDICTIONS"
        )

        print(
            "================================================"
        )


        # ----------------------------------------------------
        # 🔴 POLYMORPHISM
        #
        # Again we call:
        #
        #     predict()
        #
        # The Decision Tree provides its own implementation.
        # ----------------------------------------------------

        dt_predictions = self.decision_tree_model.predict(
            X_test
        )


        # ====================================================
        # STEP 11
        #
        # EVALUATE DECISION TREE
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 11: DECISION TREE EVALUATION"
        )

        print(
            "================================================"
        )


        dt_results = self.evaluator.evaluate(
            y_test,
            dt_predictions
        )


        # ----------------------------------------------------
        # Display Decision Tree accuracy.
        # ----------------------------------------------------

        print(
            "\nDecision Tree Accuracy:"
        )

        print(
            f"{dt_results['accuracy']:.4f}"
        )


        # ====================================================
        # STEP 12
        #
        # LOG RANDOM FOREST PARAMETERS
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 12: LOG RANDOM FOREST PARAMETERS"
        )

        print(
            "================================================"
        )


        self.mlflow_tracker.log_parameters(
            {
                "random_forest_n_estimators": 100,
                "random_forest_random_state": 42
            }
        )


        # ====================================================
        # STEP 13
        #
        # LOG RANDOM FOREST METRICS
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 13: LOG RANDOM FOREST METRICS"
        )

        print(
            "================================================"
        )


        self.mlflow_tracker.log_metrics(
            {
                "random_forest_accuracy":
                    rf_results["accuracy"]
            }
        )


        # ====================================================
        # STEP 14
        #
        # LOG DECISION TREE PARAMETERS
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 14: LOG DECISION TREE PARAMETERS"
        )

        print(
            "================================================"
        )


        self.mlflow_tracker.log_parameters(
            {
                "decision_tree_max_depth": 5
            }
        )


        # ====================================================
        # STEP 15
        #
        # LOG DECISION TREE METRICS
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 15: LOG DECISION TREE METRICS"
        )

        print(
            "================================================"
        )


        self.mlflow_tracker.log_metrics(
            {
                "decision_tree_accuracy":
                    dt_results["accuracy"]
            }
        )


        # ====================================================
        # STEP 16
        #
        # LOG RANDOM FOREST MODEL
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 16: LOG RANDOM FOREST MODEL"
        )

        print(
            "================================================"
        )


        self.mlflow_tracker.log_model(
            self.random_forest_model.model,
            artifact_path="random_forest_model"
        )


        # ====================================================
        # STEP 17
        #
        # LOG DECISION TREE MODEL
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 17: LOG DECISION TREE MODEL"
        )

        print(
            "================================================"
        )


        self.mlflow_tracker.log_model(
            self.decision_tree_model.model,
            artifact_path="decision_tree_model"
        )


        # ====================================================
        # STEP 18
        #
        # END MLFLOW RUN
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "STEP 18: END MLFLOW RUN"
        )

        print(
            "================================================"
        )


        self.mlflow_tracker.end_run()


        # ====================================================
        # STEP 19
        #
        # DISPLAY FINAL RESULTS
        # ====================================================

        print(
            "\n================================================"
        )

        print(
            "FINAL MODEL RESULTS"
        )

        print(
            "================================================"
        )


        print(
            f"\nRandom Forest Accuracy: "
            f"{rf_results['accuracy']:.4f}"
        )


        print(
            f"Decision Tree Accuracy: "
            f"{dt_results['accuracy']:.4f}"
        )


        # ====================================================
        # RETURN BOTH MODEL RESULTS
        # ====================================================
        #
        # We return a dictionary instead of only returning
        # y_test and predictions.
        #
        # This makes the result easier for main.py and
        # future MLOps components to use.
        # ====================================================

        return {
            "random_forest": rf_results,
            "decision_tree": dt_results,
            "y_test": y_test,
            "random_forest_predictions": rf_predictions,
            "decision_tree_predictions": dt_predictions
        }
