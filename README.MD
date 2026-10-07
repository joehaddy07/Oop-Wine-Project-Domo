oop-wine-quality/
│
├── .github/
│   └── workflows/
│       └── oop-wine-project.yaml
│
├── data/
│   ├── raw/
│   │   └── wine.csv
│   │
│   └── processed/
│       └── README.md
│
├── notebooks/
│   └── wine-analysis.ipynb
│
├── src/
│   ├── main.py
│   ├── pipeline.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py
│   │
│   └── mlflow/
│       ├── __init__.py
│       └── tracking.py
│
├── tests/
│   │
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_pipeline.py
│
├── artifacts/
│   │
│   ├── models/
│
├── plots/
│   └── confusion_matrix.png
│
└── reports/
│   └── classification_report.txt
│   
│   
│
├── app/
│   │
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
│
├── helm/
│   │
│   └── wine-quality/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml
│           └── service.yaml
│
├── requirements.txt
│
├── requirements-dev.txt
│
├── Makefile
│
├── README.md
│
└── .gitignore

Project structure.

========================
CI/CD interview story
=========================
I built an OOP-based Wine Quality ML application and automated the MLOps lifecycle using GitHub Actions.I set up a self-hosted GitHub Actions runner. The pipeline runs tests, trains the model, builds a Docker image tagged with the Git commit SHA, scans it for vulnerabilities with Trivy, and pushes it to Docker Hub. It then loads that exact image into a Kind Kubernetes cluster, deploys it with Helm, and verifies that Kubernetes is actually running the expected image and deployment before declaring success.



========================================================================
A structured, compelling story i can tell in an interview using the STAR method (Situation, Task, Action, Result).Debugging and fixing.
==========================================================================

Situation:
=========

"I was building a full MLOps pipeline for a wine quality classifier — GitHub Actions running tests, training a model, building a Docker image, scanning it, pushing to Docker Hub, and deploying to a Kubernetes cluster with Helm. The pipeline worked locally, but every time I pushed to CI, it failed — and each fix seemed to just reveal a new failure underneath it."

Task:
====

"I needed to get the pipeline fully green end-to-end, but more importantly, I made a deliberate choice early on: instead of patching each error as it appeared, I wanted to actually understand root causes, because I noticed the errors kept cascading — fix one, get a different one in the exact same spot."

Action:
======

"The first wall was a pandas install failure in CI that wasn't reproducing locally. I traced it by forcing pip to only accept pre-built wheels instead of silently compiling from source — that single flag turned a confusing multi-layer compiler error into one precise line: pandas 2.1.0 simply never shipped a wheel for Python 3.12. My requirements file was pinned to a version that was never installable on the Python version I'd configured — not a platform issue at all.

Later, a completely different error — MLflow failing to import a submodule — only happened in CI, never locally. I traced that one to a naming collision: I'd named my own tracking module src/mlflow, the exact same name as the real MLflow library, and depending on how the test runner resolved import paths, Python was sometimes picking up my own folder instead of the real package.

Then, once the image was actually building and deploying, Kubernetes was running the wrong image entirely. I added a verification step that compared the expected image tag against what was actually deployed, and traced it back to a secret misconfiguration — a Docker Hub access token had been stored in the field meant for the username, so the pipeline was building and pushing to a repository literally named after the token. I also caught, mid-debug, that a token had been exposed in plaintext logs, and revoked and rotated it immediately rather than just moving past it."

Result:
======

"The pipeline now runs fully clean — tests, training, build, scan, push, deploy, and a final verification step confirming Kubernetes is running the exact image from that commit. More importantly, I came out of it with a repeatable debugging habit: when an error looks environment-specific, force the tool to fail fast and loud instead of silently falling back — that one technique alone cut through three of the hardest bugs in this project."


==========================================================
How Everything Works Together
==========================================================
                Wine Dataset
                     │
                     ↓
             WineDataLoader
                     │
                     ↓
            WinePreprocessor
                     │
              ┌──────┴──────┐
              ↓             ↓
             X data         y target
              │             │
              └──────┬──────┘
                     ↓
              Train/Test Split
                     │
                     ↓
              BaseWineModel
                     │
              ┌──────┴──────┐
              ↓             ↓
        Random Forest   Decision Tree
              │             │
              └──────┬──────┘
                     ↓
                   Model
                     │
                     ↓
              ModelEvaluator
                     │
                     ↓
             Metrics / Reports
                     │
                     ↓
               WinePipeline
                     │
                     ↓
             MLOps Pipeline
                     │
          ┌──────────┼───────────┐
          ↓          ↓           ↓
       Docker    GitHub       Kubernetes
                  Actions        + Helm


“I built an OOP-based Wine Quality Prediction project and integrated it into an MLOps CI/CD pipeline.

The application starts with the wine dataset. I created a `WineDataLoader` class responsible for loading the data and a `WinePreprocessor` class responsible for creating the features and target and splitting the data into training and testing sets.

For the machine learning portion, I wanted to demonstrate good object-oriented design, so I created a `BaseWineModel` abstract class. This is where I used **abstraction**. The base class defines the common `train()` and `predict()` methods that every model must implement.

I then created `RandomForestWineModel` and `DecisionTreeWineModel` classes that inherit from `BaseWineModel`. This demonstrates **inheritance**. Each model follows the same structure but uses a different machine learning algorithm.

This also gives me **polymorphism**. Both models have the same `train()` and `predict()` interface, so the pipeline can call those methods without needing to know exactly which model is being used. The Random Forest performs Random Forest training, while the Decision Tree performs Decision Tree training.

I then created a `WinePipeline` class. This demonstrates **composition** because the pipeline contains objects such as the data loader, preprocessor, model, and evaluator. Instead of putting all the functionality into one large class, each class has a specific responsibility.

For **encapsulation**, I kept internal functionality inside the appropriate classes. For example, the evaluator handles the internal process of generating reports, metrics, and plots. Private helper methods are used for internal operations, while the rest of the application interacts with the evaluator through its public methods.

Once the OOP application was working, I connected it to the MLOps side. GitHub Actions runs the tests and training process, builds the Docker image, tags the image using the Git commit SHA, and pushes it to Docker Hub.

The deployment stage pulls the exact image, loads it into my Kind Kubernetes cluster, and uses Helm to deploy the application. The pipeline then verifies that Kubernetes is running the expected image and waits for the deployment rollout to complete.

So the project demonstrates both software engineering and MLOps. The OOP design makes the machine learning code modular and maintainable, while the CI/CD pipeline automates testing, packaging, deployment, and verification.”
