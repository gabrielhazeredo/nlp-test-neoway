# Model report - [`nlp-test-neoway`]
This report should contain all relevant information regarding your model. Someone reading this document should be able to easily understand and reproducible your findings.

## Summary

The model performs sentiment analysis on a text. It is based on regular expressions for the preprocessing and a TFIDF + SVM model for the classification. It was trained on usrs' reviews data


### Usage

Provide a reproducible pipeline to run your model

1. clone the repository
2. download test dataset from [github](https://github.com/americanas-tech/b2w-reviews01/blob/main/B2W-Reviews01.csv)
3. run the following code on command line

```python
pip install -e .
python -m nlp_test_neoway run on_my_data.csv 
```

### Output

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

### Performance Metrics
Provide any metrics used and their value in tables.

| metric    | `Yes`   | `No`      | 
| --------- | --------- | ----------- | 
| precision | .98       | .79         |
| recall    | .91       | .93         |

## Pre-processing
> Motivate each pre-processing used in one line.

1. merged 'review_title' and 'review_text': both are text data and with similiar behaviour
2. applied regex to text: clean links, numbers, special characters, double spaces, combined negations
3. Lemmatization: reduce words to their root form
4. Unidecode: remove accents from words, to prevent misspelling

## Feature selection
> Motivate any feature selection method that has been used and list/link
> the features used in the model.

* Top features on TfidfVectorizer with min_df=5, max_df=0.85: out of the 249523 features, 5000 were selected.

### Features used
'review_title' and 'review_text' : both are text data 

## Modeling
> Describe the type of model used.

A `LinearSVC` from `scikit-learn` was used. This model serves our purpose to perform binary classification and has shown satisfactory performance.

### Model selection
> Describe any model selection that has been performed.

All a `LinearSVC`, `RadomForestClassifier`, `MultinomialNB` and `LogisticRegression` have been used. The `RadomForestClassifier` model the slowest to train and overfitted in the train. We kept the `LinearSVC` for time to train, handling sparse matrix and perfromance. 

### Model validation
> Motivate your model validation strategy.

1. The dataset was divided into training and validation (80/20)
2. A stratified sampling was used to reproduce the characteristics of the deploy dataset.
3. A 2-fold strategy was used
4. See the Precision section of the model description for the achieved precision.

### Model optimization
> Motivate your choice of hyperparameters and report the training results.

* The hyper-parameters `C` and `loss` were optimized using gridsearchCV.

