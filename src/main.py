# main.py
#│
#├── Preprocess data
#├── Split data
#├── Create Random Forest
#├── Train Random Forest
#├── Predict
#├── Create Decision Tree
#├── Train Decision Tree
#├── Predict
#├── Evaluate
#├── MLflow
#└── Display results
# ============================================================
# WINE QUALITY MACHINE LEARNING APPLICATION
#
# File:
#     src/main.py
#
# PURPOSE:
#     This is the entry point of our Wine Quality ML project.
#
#     main.py should NOT contain all of our machine-learning
#     logic.
#
#     Instead, it creates a WinePipeline object and tells
#     the pipeline to run.
#
#
# OOP CONCEPTS:
#
#     🟡 COMPOSITION
#         WinePipeline contains:
#
#             WineDataLoader
#             WinePreprocessor
#             RandomForestWineModel
#             DecisionTreeWineModel
#             ModelEvaluator
#             MLflowTracker
#
#
#     🔵 ABSTRACTION
#         main.py does not need to know how:
#
#             data is loaded
#             data is processed
#             models are trained
#             predictions are made
#             models are evaluated
#             MLflow is used
#
#         main.py simply calls:
#
#             pipeline.run()
#
#
#     🟣 ENCAPSULATION
#         The internal ML workflow is hidden inside
#         WinePipeline and the classes it contains.
# ============================================================


# ============================================================
# IMPORT WINE PIPELINE
# ============================================================
#
# WinePipeline coordinates the complete machine-learning
# workflow.
#
# ============================================================

from pipeline import WinePipeline


# ============================================================
# MAIN FUNCTION
# ============================================================
#
# Instead of writing all of our ML code directly in this file,
# we place the application startup logic inside main().
#
# ============================================================

def main():

    # ========================================================
    # STEP 1
    #
    # Define the location of our wine dataset.
    # ========================================================

    data_path = "data/raw/wine.csv"


    # ========================================================
    # STEP 2
    #
    # Create the WinePipeline object.
    #
    # 🟡 COMPOSITION
    #
    # When we create WinePipeline, its __init__() creates:
    #
    #     WineDataLoader
    #     WinePreprocessor
    #     RandomForestWineModel
    #     DecisionTreeWineModel
    #     ModelEvaluator
    #     MLflowTracker
    #
    # ========================================================

    pipeline = WinePipeline(
        data_path
    )


    # ========================================================
    # STEP 3
    #
    # Run the complete machine-learning pipeline.
    #
    # 🔵 ABSTRACTION
    #
    # We don't need to know all the steps happening inside
    # pipeline.run().
    #
    # We simply tell the pipeline:
    #
    #     "Run the ML workflow."
    #
    # ========================================================

    results = pipeline.run()


    # ========================================================
    # STEP 4
    #
    # Display the final results.
    #
    # pipeline.run() returns a dictionary containing the
    # results for both models.
    # ========================================================

    print(
        "\n================================================"
    )

    print(
        "FINAL RESULTS FROM MAIN.PY"
    )

    print(
        "================================================"
    )


    # ========================================================
    # RANDOM FOREST RESULTS
    # ========================================================

    random_forest_results = results[
        "random_forest"
    ]


    print(
        "\nRandom Forest Accuracy:"
    )

    print(
        f"{random_forest_results['accuracy']:.4f}"
    )


    # ========================================================
    # DECISION TREE RESULTS
    # ========================================================

    decision_tree_results = results[
        "decision_tree"
    ]


    print(
        "\nDecision Tree Accuracy:"
    )

    print(
        f"{decision_tree_results['accuracy']:.4f}"
    )


    # ========================================================
    # DISPLAY RANDOM FOREST PREDICTIONS
    # ========================================================

    print(
        "\nRandom Forest Predictions:"
    )

    print(
        results[
            "random_forest_predictions"
        ]
    )


    # ========================================================
    # DISPLAY DECISION TREE PREDICTIONS
    # ========================================================

    print(
        "\nDecision Tree Predictions:"
    )

    print(
        results[
            "decision_tree_predictions"
        ]
    )


    # ========================================================
    # DISPLAY COMPLETION MESSAGE
    # ========================================================

    print(
        "\n================================================"
    )

    print(
        "Wine Quality ML Pipeline Completed Successfully!"
    )

    print(
        "================================================"
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================
#
# This condition checks whether Python is executing this file
# directly.
#
# If we run:
#
#     python src/main.py
#
# then __name__ becomes "__main__".
#
# Therefore main() will execute.
#
# ============================================================

if __name__ == "__main__":

    main()
