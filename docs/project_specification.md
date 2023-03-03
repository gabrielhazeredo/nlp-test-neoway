# Project specification - [`nlp-test-neoway`]
> This document contains a data-science oriented description of the project. It orients the data science project towards data and technology. It serves the purposes of

* outlining a solution,
* defining the scope of the project,
* defining measurable success metrics,
* defining interfaces to the production environment,
* gathering information regarding relevant datasets and features to the problem,
* upstream communication and agreement on requisites.


## Checklist
> Mark which tasks have been performed.

- [ ] **Summary**: all major specs are defined including success criteria, scope, etc.
- [ ] **Output specification**: the output format, metadata and indexation have been defined and agreed upon.
- [ ] **Solution architecture**: a viable and minimal architecture for the final solution has been provided and agreed upon.
- [ ] **Limitations and risks**: a list with the main limitations and risks associated to the project has been provided.
- [ ] **Related resources**: lists with related datasets, features and past projects have been given.
- [ ] **Peer-review**: you discussed and brainstormed with colleagues the outlined specifications.
- [ ] **Acceptance**: these specifications have been accepted by the Data and Product directors. State names and date.

## Summary
> The table below summarizes the key requirements for the project.

| problem type              | target population | entity | N_target | N_labeled | sucess_metrics | updt_freq |
|---------------------------|-------------------|--------|----------|-----------|----------------|-----------|
| binary classification     | online reviewers  | reviewers' ids    | 130K     | 130K      | precision      | monthly   |


### Objective
> Provide a short (max 3-line) description  of the project's objective.

"This project aims at developing a model to automatically determine if the reviewer would recommend the product."

### Target population
> More detailed description of the population to which the model should apply. Include any relevant characteristics.

| Entity          | Region      | Type             | N_target |
|-----------------|-------------|------------------|----------|
| reviewers' ids  | Brasil      | online reviewers | 30M      |
                                    |

### Output specification
> Describe how the output of the model will be delivered, including its domain and metadata.

The model outputs the sentiment of the text, based on the probability of the text leading to a recommendation of the product. The output is a string with the following values: `Yes` `No`.

```python
output_example1 = ['Yes'] 
```

#### Metadata

* a brief description: this model predicts the type of a restaurant
  cuisine
* a measure of accuracy applicable for use of the model in other
  setups (if applicable): precision, error matrix.
* model version: 1.0
* author: Gabriel Hartmann de Azeredo
* date created: 03/03/2023
* link to training data: https://github.com/americanas-tech/b2w-reviews01/blob/main/B2W-Reviews01.csv

Make sure that the final consumer of your model can make use of your metadata.

### Problem type
> Describe to which Data Science problem type(s) this project relates to with a brief motivation.

"Since the objective is to assign one label to a text, this problem is a binary classification. It is also supervised since observed data is available."

### Limitations and risks
> Provide a list with the main limitations and the associated risks for this project. (lile are supposed to be a well-educated guess)

| Limitation                              | Likelihood | Loss                               | Contingency                        |
|-----------------------------------------|------------|------------------------------------|------------------------------------|
| Very imbalanced dataset                 | 100%       | lack of data for label `No`        | Use precision as metric, weigth loss (change threshold if possible)|
