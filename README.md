# [`nlp-test-neoway`]
> This project consists on the NLP Test from Neoway for the Data Scientist position. Its goal is to create a model that can classify a text based on its sentiment. This model uses the B2W-Reviews01 dataset, which contains features and data descriptive of product reviews and with potential predictive capabilities

This model is a `NLP SVM model` that `performs sentiment analysis` to solve `automatic recommendation based text`.

## Stakeholders
> Describe the people involved in this project

| Role                 | Responsibility         | Full name                | e-mail       |
| -----                | ----------------       | -----------              | ---------    |
| Data Scientist       | Author                 | [`Gabriel Hartmann de Azeredo`]            | [`gabriel.hazeredo@gmail.com`] |


## Usage
> How to reproduce the model

Usage is standardized across models. There are two main things you need to know, the development workflow and the Makefile commands.

Both are made super simple to work with Git and Docker while versioning experiments and workspace.

All you'll need to have setup is Docker and Git, which you probably already have. If you don't, feel free to ask for help.

Makefile commands can be accessed using `make help`.

Make sure that **docker** is installed.

Clone the project from the repo.
```
git clone https://github.com/gabrielhazeredo/nlp-test-neoway.git
cd nlp-test-neoway
```


## Final Report (to be filled once the project is done)

### Model Frequency

> Prediction time is  ~ 3 seconds for single prediction mode

### Model updating

> For simple retraining, the model can be updated by running the `make train` command. This will retrain the model with the latest data and save the model in the `models` folder. Changes in the pipeline can be made by editing the `nlp_test_neoway/main.py` file.

### Maintenance

> Describe how your model may be maintained in the future

### Minimum viable product

> Any sort of API that provide data as a JSON object with the following structure:

```
{
    'review_title': 'Title of the review to be classified', 
    'review_text': 'Body text of the review to be classified'
}
```

## Documentation

* [project_specification.md](./docs/project_specification.md): gives a data-science oriented description of the project.

* [model_report.md](./docs/model_report.md): describes the modeling performed.


#### Folder structure
>Explain you folder strucure

* [docs](./docs): contains documentation of the project
* [analysis](./analysis/): contains notebooks of data and modeling experimentation.
* [tests](./tests/): contains files used for unit tests.
* [nlp_test_neoway](./nlp_test_neoway/): main Python package with source of the model.
