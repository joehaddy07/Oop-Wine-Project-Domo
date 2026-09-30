# ============================================================
# MODULE: src/mlflow/tracking.py
#
# PURPOSE:
#     Manage all MLflow tracking operations for our
#     Wine Quality Machine Learning project.
#
#
# OOP CONCEPTS:
#
#     🟣 ENCAPSULATION
#         This class hides the details of MLflow operations.
#
#     🟡 COMPOSITION
#         WinePipeline will contain an MLflowTracker object.
#
#     🔵 ABSTRACTION
#         WinePipeline will simply tell the tracker:
#
#             log_parameters()
#             log_metrics()
#             log_model()
#             log_artifacts()
#
#         It does not need to know how MLflow performs
#         those operations internally.
#
#     🟢 INHERITANCE
#         No inheritance is required for this class.
#
#     🔴 POLYMORPHISM
#         No direct polymorphism is required for this class.
# ============================================================


# ============================================================
# IMPORT MLflow
# ============================================================

# ------------------------------------------------------------
# Import the main MLflow library.
#
# We use MLflow to track:
#
#     Experiments
#     Runs
#     Parameters
#     Metrics
#     Models
#     Artifacts
# ------------------------------------------------------------

import mlflow


# ------------------------------------------------------------
# Import MLflow's sklearn integration.
#
# This allows us to save a Scikit-Learn model directly
# into MLflow.
# ------------------------------------------------------------

import mlflow.sklearn


# ============================================================
# CREATE MLflowTracker CLASS
# ============================================================

class MLflowTracker:

    """
    MLflowTracker manages MLflow tracking for our
    Wine Quality Machine Learning project.

    🟣 ENCAPSULATION:

    The details of MLflow are hidden inside this class.

    Other parts of our application do not need to directly
    call mlflow.start_run(), mlflow.log_metric(), etc.

    They can simply call methods such as:

        tracker.start_run()
        tracker.log_parameters()
        tracker.log_metrics()
        tracker.log_model()
        tracker.log_artifact()
        tracker.end_run()
    """


    # ========================================================
    # CONSTRUCTOR
    # ========================================================
    #
    # __init__ runs automatically when we create:
    #
    #     tracker = MLflowTracker()
    #
    # ========================================================

    def __init__(
        self,
        tracking_uri="http://localhost:5000",
        experiment_name="Wine_Quality_Predictions"
    ):

        # ----------------------------------------------------
        # Store the MLflow server location.
        #
        # Example:
        #
        #     http://localhost:5000
        #
        # This tells MLflow where our tracking server is.
        # ----------------------------------------------------

        self.tracking_uri = tracking_uri


        # ----------------------------------------------------
        # Store the name of our MLflow experiment.
        # ----------------------------------------------------

        self.experiment_name = experiment_name


        # ----------------------------------------------------
        # Configure MLflow to use our tracking server.
        # ----------------------------------------------------

        mlflow.set_tracking_uri(
            self.tracking_uri
        )


        # ----------------------------------------------------
        # Tell MLflow which experiment we want to use.
        #
        # If the experiment does not already exist,
        # MLflow will create it.
        # ----------------------------------------------------

        mlflow.set_experiment(
            self.experiment_name
        )


        # ----------------------------------------------------
        # Display information so we know which MLflow server
        # our application is using.
        # ----------------------------------------------------

        print(
            f"[INFO] MLflow tracking URI: "
            f"{self.tracking_uri}"
        )


        print(
            f"[INFO] MLflow experiment: "
            f"{self.experiment_name}"
        )


    # ========================================================
    # START MLflow RUN
    # ========================================================
    #
    # This method starts a new MLflow run.
    #
    # A RUN represents one execution of our ML pipeline.
    #
    # Example:
    #
    #     Run 1
    #     Run 2
    #     Run 3
    #
    # Each run can contain different:
    #
    #     Parameters
    #     Metrics
    #     Models
    #     Artifacts
    #
    # ========================================================

    def start_run(
        self,
        run_name="OOP_Wine_Quality_Model"
    ):

        # ----------------------------------------------------
        # Start the MLflow run.
        # ----------------------------------------------------

        mlflow.start_run(
            run_name=run_name
        )


        # ----------------------------------------------------
        # Display confirmation.
        # ----------------------------------------------------

        print(
            f"[INFO] MLflow run started: "
            f"{run_name}"
        )


    # ========================================================
    # LOG PARAMETERS
    # ========================================================
    #
    # Parameters are configuration values used during
    # model training.
    #
    # Examples:
    #
    #     n_estimators = 100
    #     random_state = 42
    # ========================================================

    def log_parameters(
        self,
        parameters
    ):

        # ----------------------------------------------------
        # Loop through the dictionary containing parameters.
        #
        # Example:
        #
        # {
        #     "n_estimators": 100,
        #     "random_state": 42
        # }
        # ----------------------------------------------------

        for name, value in parameters.items():

            # ------------------------------------------------
            # Log each parameter to MLflow.
            # ------------------------------------------------

            mlflow.log_param(
                name,
                value
            )


        # ----------------------------------------------------
        # Display confirmation.
        # ----------------------------------------------------

        print(
            "[INFO] MLflow parameters logged."
        )


    # ========================================================
    # LOG METRICS
    # ========================================================
    #
    # Metrics are numerical measurements of model performance.
    #
    # Examples:
    #
    #     accuracy
    #     precision
    #     recall
    #     f1-score
    #
    # ========================================================

    def log_metrics(
        self,
        metrics
    ):

        # ----------------------------------------------------
        # Loop through our metrics dictionary.
        #
        # Example:
        #
        # {
        #     "accuracy": 0.87
        # }
        # ----------------------------------------------------

        for name, value in metrics.items():

            # ------------------------------------------------
            # MLflow expects numerical values for metrics.
            # ------------------------------------------------

            mlflow.log_metric(
                name,
                float(value)
            )


        # ----------------------------------------------------
        # Display confirmation.
        # ----------------------------------------------------

        print(
            "[INFO] MLflow metrics logged."
        )


    # ========================================================
    # LOG MODEL
    # ========================================================
    #
    # Save our trained Scikit-Learn model into MLflow.
    #
    # ========================================================

    def log_model(
        self,
        model,
        artifact_path="model"
    ):

        # ----------------------------------------------------
        # Save the Scikit-Learn model to MLflow.
        #
        # mlflow.sklearn understands Scikit-Learn models.
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path
        )


        # ----------------------------------------------------
        # Display confirmation.
        # ----------------------------------------------------

        print(
            "[INFO] Model logged to MLflow."
        )


    # ========================================================
    # LOG ARTIFACT
    # ========================================================
    #
    # Artifacts are files generated during our ML process.
    #
    # Examples:
    #
    #     confusion_matrix.png
    #     classification_report.txt
    #     wine_quality.pkl
    #
    # ========================================================

    def log_artifact(
        self,
        file_path,
        artifact_path=None
    ):

        # ----------------------------------------------------
        # Send the file to MLflow.
        # ----------------------------------------------------

        mlflow.log_artifact(
            file_path,
            artifact_path=artifact_path
        )


        # ----------------------------------------------------
        # Display confirmation.
        # ----------------------------------------------------

        print(
            f"[INFO] Artifact logged: "
            f"{file_path}"
        )


    # ========================================================
    # END MLflow RUN
    # ========================================================
    #
    # This method closes the current MLflow run.
    #
    # ========================================================

    def end_run(self):

        # ----------------------------------------------------
        # End the active MLflow run.
        # ----------------------------------------------------

        mlflow.end_run()


        # ----------------------------------------------------
        # Display confirmation.
        # ----------------------------------------------------

        print(
            "[INFO] MLflow run completed."
        )
