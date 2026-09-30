#Finally, we test the entire pipeline.This is our integration-style test:
#========================================================================
#Data
 #↓
#Preprocessing
 #↓
#Random Forest
 #↓
#Decision Tree
 #↓
#Evaluation
 #↓
#MLflow
#========================================================================

# ============================================================
# TEST COMPLETE WINE PIPELINE
#
# File:
#     tests/test_pipeline.py
#
# PURPOSE:
#     Test the complete WinePipeline.
#
#
# OOP CONCEPT:
#
#     🟡 COMPOSITION
#
#     WinePipeline contains:
#
#         WineDataLoader
#         WinePreprocessor
#         RandomForestWineModel
#         DecisionTreeWineModel
#         ModelEvaluator
#         MLflowTracker
# ============================================================


# ============================================================
# IMPORT WINE PIPELINE
# ============================================================

from src.pipeline import WinePipeline


# ============================================================
# TEST COMPLETE PIPELINE
# ============================================================

def test_wine_pipeline():

    # --------------------------------------------------------
    # Create the pipeline.
    #
    # 🟡 COMPOSITION
    #
    # Creating WinePipeline creates all of its internal
    # component objects.
    # --------------------------------------------------------

    pipeline = WinePipeline(
        "data/raw/wine.csv"
    )


    # --------------------------------------------------------
    # Run the complete pipeline.
    # --------------------------------------------------------

    results = pipeline.run()


    # ========================================================
    # VERIFY RESULT STRUCTURE
    # ========================================================

    # --------------------------------------------------------
    # Make sure results is a dictionary.
    # --------------------------------------------------------

    assert isinstance(
        results,
        dict
    )


    # --------------------------------------------------------
    # Verify Random Forest results exist.
    # --------------------------------------------------------

    assert "random_forest" in results


    # --------------------------------------------------------
    # Verify Decision Tree results exist.
    # --------------------------------------------------------

    assert "decision_tree" in results


    # --------------------------------------------------------
    # Verify predictions exist.
    # --------------------------------------------------------

    assert "random_forest_predictions" in results

    assert "decision_tree_predictions" in results


    # --------------------------------------------------------
    # Verify y_test exists.
    # --------------------------------------------------------

    assert "y_test" in results


    # ========================================================
    # VERIFY MODEL METRICS
    # ========================================================

    # --------------------------------------------------------
    # Get Random Forest accuracy.
    # --------------------------------------------------------

    rf_accuracy = results[
        "random_forest"
    ][
        "accuracy"
    ]


    # --------------------------------------------------------
    # Get Decision Tree accuracy.
    # --------------------------------------------------------

    dt_accuracy = results[
        "decision_tree"
    ][
        "accuracy"
    ]


    # --------------------------------------------------------
    # Accuracy should be between 0 and 1.
    # --------------------------------------------------------

    assert 0 <= rf_accuracy <= 1

    assert 0 <= dt_accuracy <= 1
