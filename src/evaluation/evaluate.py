
# ============================================================
# WINE MODEL EVALUATION
#
# File:
#     src/evaluation/evaluate.py
#
# PURPOSE:
#     Evaluate the performance of our machine-learning model
#     and save the evaluation results as artifacts.
#
# RESPONSIBILITIES:
#
#     1. Calculate accuracy
#     2. Generate classification report
#     3. Generate confusion matrix
#     4. Save classification report
#     5. Save confusion matrix plot
#
#
# OOP CONCEPTS:
#
#     🟣 ENCAPSULATION
#
#     This class hides the details of:
#
#         - calculating metrics
#         - creating directories
#         - creating file names
#         - saving reports
#         - creating confusion matrix plots
#
#     The rest of the application simply calls:
#
#         evaluator.evaluate()
#
#
#     🟡 COMPOSITION
#
#     WinePipeline HAS-A ModelEvaluator.
#
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

# ------------------------------------------------------------
# Import os.
#
# We use os to create directories and build file paths.
# ------------------------------------------------------------

import os


# ------------------------------------------------------------
# Import matplotlib.
#
# We use matplotlib to create the confusion matrix image.
# ------------------------------------------------------------

import matplotlib

# ------------------------------------------------------------
# "Agg" allows matplotlib to work in environments without
# a graphical display.
#
# This is especially useful for:
#
#     Docker
#     GitHub Actions
#     Kubernetes
#     CI/CD servers
# ------------------------------------------------------------

matplotlib.use("Agg")


# ------------------------------------------------------------
# Import pyplot.
# ------------------------------------------------------------

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Import seaborn.
#
# Seaborn gives us a nicer-looking heatmap.
# ------------------------------------------------------------

import seaborn as sns


# ------------------------------------------------------------
# Import accuracy_score.
#
# This calculates the percentage of predictions that were
# correct.
# ------------------------------------------------------------

from sklearn.metrics import accuracy_score


# ------------------------------------------------------------
# Import classification_report.
#
# This provides:
#
#     precision
#     recall
#     f1-score
# ------------------------------------------------------------

from sklearn.metrics import classification_report


# ------------------------------------------------------------
# Import confusion_matrix.
#
# This shows the relationship between:
#
#     Actual values
#     Predicted values
# ------------------------------------------------------------

from sklearn.metrics import confusion_matrix


# ============================================================
# MODEL EVALUATOR CLASS
# ============================================================

class ModelEvaluator:

    """
    ModelEvaluator evaluates the performance of a machine-
    learning model and saves evaluation artifacts.
    """


    # ========================================================
    # CONSTRUCTOR
    # ========================================================
    #
    # The constructor receives:
    #
    #     output_dir
    #     model_name
    #
    # Example:
    #
    #     ModelEvaluator(
    #         output_dir="./artifacts",
    #         model_name="random_forest"
    #     )
    #
    # ========================================================

    def __init__(
        self,
        output_dir="./artifacts",
        model_name="model"
    ):

        # ----------------------------------------------------
        # Store the main artifact directory.
        #
        # Example:
        #
        #     ./artifacts
        # ----------------------------------------------------

        self.output_dir = output_dir


        # ----------------------------------------------------
        # Store the model name.
        #
        # Example:
        #
        #     random_forest
        #
        # or:
        #
        #     decision_tree
        # ----------------------------------------------------

        self.model_name = model_name


        # ====================================================
        # 🟣 ENCAPSULATION
        #
        # Create the directories internally.
        #
        # The caller does not need to worry about whether
        # these directories already exist.
        # ====================================================

        self.reports_dir = os.path.join(
            self.output_dir,
            "reports"
        )


        self.plots_dir = os.path.join(
            self.output_dir,
            "plots"
        )


        # ----------------------------------------------------
        # Create the directories.
        #
        # exist_ok=True means:
        #
        #     If the directory already exists,
        #     do nothing.
        #
        #     If it does not exist,
        #     create it.
        # ----------------------------------------------------

        os.makedirs(
            self.reports_dir,
            exist_ok=True
        )


        os.makedirs(
            self.plots_dir,
            exist_ok=True
        )


    # ========================================================
    # EVALUATE MODEL
    # ========================================================
    #
    # This is the main public method.
    #
    # The rest of the application calls:
    #
    #     evaluator.evaluate(
    #         y_test,
    #         predictions
    #     )
    #
    # ========================================================

    def evaluate(
        self,
        y_test,
        predictions
    ):

        # ----------------------------------------------------
        # Display which model we are evaluating.
        # ----------------------------------------------------

        print(
            f"\n[INFO] Evaluating "
            f"{self.model_name}..."
        )


        # ====================================================
        # STEP 1
        #
        # CALCULATE ACCURACY
        # ====================================================

        # ----------------------------------------------------
        # accuracy_score compares:
        #
        #     actual values
        #
        # against:
        #
        #     predicted values
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )


        # ====================================================
        # STEP 2
        #
        # GENERATE CLASSIFICATION REPORT
        # ====================================================

        # ----------------------------------------------------
        # Generate:
        #
        #     precision
        #     recall
        #     f1-score
        #     support
        # ----------------------------------------------------

        report = classification_report(
            y_test,
            predictions
        )


        # ====================================================
        # STEP 3
        #
        # GENERATE CONFUSION MATRIX
        # ====================================================

        confusion = confusion_matrix(
            y_test,
            predictions
        )


        # ====================================================
        # STEP 4
        #
        # SAVE CLASSIFICATION REPORT
        # ====================================================

        self._save_classification_report(
            report
        )


        # ====================================================
        # STEP 5
        #
        # SAVE CONFUSION MATRIX
        # ====================================================

        self._save_confusion_matrix(
            confusion
        )


        # ====================================================
        # STEP 6
        #
        # RETURN RESULTS
        # ====================================================

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": confusion
        }


    # ========================================================
    # PRIVATE METHOD:
    # SAVE CLASSIFICATION REPORT
    # ========================================================
    #
    # 🟣 ENCAPSULATION
    #
    # The underscore "_" tells us this method is intended
    # for internal use by ModelEvaluator.
    #
    # The pipeline does NOT need to call this directly.
    #
    # ========================================================

    def _save_classification_report(
        self,
        report
    ):

        # ----------------------------------------------------
        # Create a unique file name using the model name.
        #
        # Example:
        #
        #     random_forest_classification_report.txt
        #
        #     decision_tree_classification_report.txt
        # ----------------------------------------------------

        file_name = (
            f"{self.model_name}_"
            f"classification_report.txt"
        )


        # ----------------------------------------------------
        # Build the complete file path.
        # ----------------------------------------------------

        file_path = os.path.join(
            self.reports_dir,
            file_name
        )


        # ----------------------------------------------------
        # Open the file in write mode.
        # ----------------------------------------------------

        with open(
            file_path,
            "w"
        ) as file:

            # -----------------------------------------------
            # Write the classification report into the file.
            # -----------------------------------------------

            file.write(
                report
            )


        # ----------------------------------------------------
        # Display confirmation.
        # ----------------------------------------------------

        print(
            f"[INFO] Classification report saved: "
            f"{file_path}"
        )


    # ========================================================
    # PRIVATE METHOD:
    # SAVE CONFUSION MATRIX
    # ========================================================
    #
    # 🟣 ENCAPSULATION
    #
    # This method hides the matplotlib/seaborn details.
    #
    # The pipeline simply calls:
    #
    #     evaluate()
    #
    # It does not need to know how the image is created.
    #
    # ========================================================

    def _save_confusion_matrix(
        self,
        confusion
    ):

        # ----------------------------------------------------
        # Create a new matplotlib figure.
        # ----------------------------------------------------

        plt.figure(
            figsize=(6, 5)
        )


        # ----------------------------------------------------
        # Create the confusion matrix heatmap.
        # ----------------------------------------------------

        sns.heatmap(
            confusion,
            annot=True,
            fmt="d",
            cmap="Blues"
        )


        # ----------------------------------------------------
        # Add a title containing the model name.
        # ----------------------------------------------------

        plt.title(
            f"{self.model_name} Confusion Matrix"
        )


        # ----------------------------------------------------
        # Create a unique file name.
        #
        # Example:
        #
        #     random_forest_confusion_matrix.png
        #
        #     decision_tree_confusion_matrix.png
        # ----------------------------------------------------

        file_name = (
            f"{self.model_name}_"
            f"confusion_matrix.png"
        )


        # ----------------------------------------------------
        # Build the complete file path.
        # ----------------------------------------------------

        file_path = os.path.join(
            self.plots_dir,
            file_name
        )


        # ----------------------------------------------------
        # Save the image.
        # ----------------------------------------------------

        plt.savefig(
            file_path,
            bbox_inches="tight"
        )


        # ----------------------------------------------------
        # Close the figure.
        #
        # This is important when generating multiple plots.
        # ----------------------------------------------------

        plt.close()


        # ----------------------------------------------------
        # Display confirmation.
        # ----------------------------------------------------

        print(
            f"[INFO] Confusion matrix saved: "
            f"{file_path}"
        )
