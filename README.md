# MLOps End to End with Model Monitoring Template

This repo is intended to demonstrate an end-to-end ML experimentation and MLOps workflow on Databricks, where a model trained in development environment, and is then deployed in production thanks to automated workflows in Databricks.


The use case at hand is a home credit risk classification. We use the [Home Credit Risk Dataset](https://www.kaggle.com/competitions/home-credit-default-risk) to build a simple classifier to predict whether a home credit is at risk of being repaid or not.

## Pipelines

Each pipeline (e.g model training pipeline, model deployment pipeline) is deployed as a [Databricks job](https://docs.databricks.com/data-engineering/jobs/jobs.html), where these jobs are deployed to a Databricks workspace using Databricks Labs' [`dbx`](https://dbx.readthedocs.io/en/latest/index.html) tool. 

The following pipelines currently defined within the package can be found at [`deployment.yml`](https://github.com/julie-nguyen-ds/telkomsel-home-credit-risk/blob/main/conf/deployment.yml) and are:
- `DEV-home-credit-setup`
    - Deletes existing feature store delta files (feature store table has to be deleted manually, **TO AUTOMATE**), existing MLflow experiments and models registered to MLflow Model Registry, in order to start afresh in the development environment.  
- `PROD-home-credit-setup`
    - Deletes existing feature store delta files (feature store table has to be deleted manually, **TO AUTOMATE**), existing MLflow experiments and models registered to MLflow Model Registry, in order to start afresh in the production environment.  
- `PROD-external-featurization`
    - Create a new feature store table and adds some features. This table is considered to already exist for collaboration and be used to browse and add features on top.
- `PROD-home-credit-ml-training`
    - Add new features to the existing feature store table.
    - Trains a light GBM model and register the model as `Staging`.
    - Compare the Staging versus Production model in the MLflow Model Registry. Transition the Staging model to Production if outperforming the current Production model.
    - Generates data quality and model degradation reports.
    
## Demo
The following outlines the workflow to demo the repo.

### Set up
1. Fork the repository [https://github.com/julie-nguyen-ds/telkomsel-home-credit-risk]
1. Configure [Databricks CLI connection profile](https://docs.databricks.com/dev-tools/cli/index.html#connection-profiles)
    - The project is designed to use 2 different Databricks CLI connection profiles: dev and prod. 
      These profiles are set in [telkomsel-home-credit-risk/.dbx/project.json](https://github.com/julie-nguyen-ds/telkomsel-home-credit-risk/blob/main/.dbx/project.json).
    - This [project.json](https://github.com/julie-nguyen-ds/telkomsel-home-credit-risk/blob/main/.dbx/project.json) file will have to be 
      adjusted accordingly to the connection profiles a user has configured on their local machine.
    - Make sure your version of `dbx` is up-to-date
1. Configure Databricks secrets for GitHub Actions (ensure GitHub actions are enabled for you forked project, as the default is off in a forked repo).
    - Within the GitHub project navigate to Secrets under the project settings
    - To run the GitHub actions workflows we require the following GitHub actions secrets:
        - `DATABRICKS_STAGING_HOST`
            - URL of Databricks dev workspace
        - `DATABRICKS_STAGING_TOKEN`
            - [Databricks access token](https://docs.databricks.com/dev-tools/api/latest/authentication.html) for dev workspace
        - `DATABRICKS_PROD_HOST`
            - URL of Databricks production workspace
        - `DATABRICKS_PROD_TOKEN`
            - [Databricks access token](https://docs.databricks.com/dev-tools/api/latest/authentication.html) for production workspace
        - `GH_TOKEN`
            - GitHub [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

    #### ASIDE: Starting from scratch
    The following resources should not be present if starting from scratch: 
    - Feature table must be deleted (and manually delete the remaining table too)
        - The table will be created when the environment setup pipeline is triggered.
    - MLflow experiment
        - MLflow Experiments during model training and model deployment will be used in both the dev and prod environments. 
          The paths to these experiments are configured in [conf/deployment.yml](https://github.com/niall-turbitt/e2e-mlops/blob/main/conf/deployment.yml).
        - For demo purposes, we delete these experiments if they exist to begin from a blank slate.
    - Model Registry
        - Delete Model in MLflow Model Registry if exists.
    
    **NOTE:** As part of the `initial-model-train-register` multitask job, the first task `demo-setup` will delete these, 
   as specified in [`demo_setup.yml`](https://github.com/niall-turbitt/e2e-mlops/blob/main/conf/job_configs/demo_setup.yml).

### Workflow

1. **Run the notebooks in development environment to have a regular ML experiment experience.
2. **Run the setup pipeline in production and external featurization pipeline (initialization of the production environment)
    - To demonstrate a CICD workflow, we want to start from a “steady state” where there is a current model in production. 
      As such, we will manually trigger the setup job to do the following steps:
      1. Set up the workspace for the demo by deleting existing MLflow experiments and register models, along with 
         existing Feature Store and labels tables. 
      1. Create a new Feature Store table to be used by the model training pipeline.
    - Run the model training and deployment pipeline to train a first baseline model to put in production
    - Manually promote this newly trained model to production via the MLflow Model Registry UI.
    - You can now do the next steps to experience CI/CD and MLOps lifecycle.


3. **Code change / model update (Continuous Integration)**

    - Create new “dev/new_model” branch 
        - `git checkout -b  dev/new_model`
    - Make a change to the [`model_train.yml`](https://github.com/niall-turbitt/e2e-mlops/blob/main/conf/job_configs/model_train.yml) config file, updating `max_depth` under model_params from 4 to 8
        - Optional: change run name under mlflow params in [`model_train.yml`](https://github.com/niall-turbitt/e2e-mlops/blob/main/conf/job_configs/model_train.yml) config file
    - Create pull request, to instantiate a request to merge the branch dev/new_model into main. 

* On pull request the following steps are triggered in the GitHub Actions workflow [`onpullrequest.yml`](https://github.com/julie-nguyen-ds/telkomsel-home-credit-risk/blob/main/.github/workflows/onpullrequest.yml):
    1. Trigger unit tests 
* Note that upon tests successfully passing, this merge request will have to be confirmed in GitHub.    


4. **Cut release**

    - Create tag (e.g. `v0.0.1`)
        - `git tag <tag_name> -a -m “Message”`
            - Note that tags are matched to `v*`, i.e. `v1.0`, `v20.15.10`
    - Push tag
        - `git push origin <tag_name>`

    - On pushing this the following steps are triggered in the [`onrelease.yml`](https://github.com/julie-nguyen-ds/telkomsel-home-credit-risk/blob/main/.github/workflows/onrelease.yml) GitHub Actions workflow:
        1. Trigger unit tests.
        1. Deploy environment setup job to the prod environment.
        1. Deploy external featurization job to the prod environment.
        1. Deploy model training and deployment job to the prod environment.
        2. Run model training and deployment job to the prod environment.

            - These jobs will now all be present in the specified workspace, and visible under the [Workflows](https://docs.databricks.com/data-engineering/jobs/index.html) tab.
    

## Limitations
- Multitask jobs running against the same cluster
    - The pipeline initial-model-train-register is a [multitask job](https://docs.databricks.com/data-engineering/jobs/index.html) 
      which stitches together demo setup, feature store creation and model train pipelines. 
    - At present, each of these tasks within the multitask job is executed on a different automated job cluster, 
      rather than all tasks executed on the same cluster. As such, there will be time incurred for each task to acquire 
      cluster resources and install dependencies.
    - As above, we recommend using a pool from which instances can be acquired when jobs are launched to reduce cluster start up time.
    
---
## Development

While using this project, you need Python 3.X and `pip` or `conda` for package management.

### Installing project requirements

```bash
pip install -r unit-requirements.txt
```

### Install project package in a developer mode

```bash
pip install -e .
```

### Testing

#### Running unit tests

For unit testing, please use `pytest`:
```
pytest tests/unit --cov
```

Please check the directory `tests/unit` for more details on how to use unit tests.
In the `tests/unit/conftest.py` you'll also find useful testing primitives, such as local Spark instance with Delta support, local MLflow and DBUtils fixture.
