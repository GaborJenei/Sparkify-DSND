# Sparkify-DSND

Sparkify is an imaginary popular music streaming service similar to Spotify or Pandora.
Millions of users stream music daily using either the free tier subscription with advertisements placed in-between songs or the monthly subscription package with no advertisements.
Users can upgrade, downgrade or cancel anytime, so it's crucial they like the service.
Every interaction with the service (play a song, like, login, logout, upgrade, etc.) is recorded, hoping that this dataset will help us gain some insights.
The overall goal of this project is to try to identify users before they leave so they can be offered different incentives increasing the chance of not cancelling the service.

A blog post of this project can be found on [medium](https://gaborjenei.medium.com/predicting-churn-with-spark-7e212d65cfb1).

# Project Background
## Project Overview, Definition
Sparkify music streaming service has two subscription tiers, a paid level with no ads and a free tier with ads playing between songs. Therefore, it is detrimental to Sparkify's bottom line to lose users by complete cancellation.
This project's primary objective is to identify users cancelling the service.
The dataset for this is an approx 12GB log file holding all user interactions in JSON format.

## Problem Statement
The way Sparkify generates revenue in two main ways:
 1. Users pay a monthly fee for the Upgraded subscription tier, where users will have no advertisements
 2. Users use the free subscription tier paying no monthly fee. However, they will have advertisements placed in-between songs. In this case, advertising partners will pay a fee for playing their ad to our users.

If a user leaves Sparkify by cancelling the service, then none of the two monetisation methods will apply, and the total turnover of Sparkify will be less.

Based on the data provided, 22% per cent of all users cancels the service, and as an approximation, we can assume that this means the same proportion reduction in the turnover.

Sparkify is eager to identify users likely to churn before they leave.
They will be able to offer them perks in the hope of preventing them from leaving, increasing the revenue and profit of the company.

The expected solution is a binary classification machine learning model. This model can be used to continuously predict if a user is likely to churn or not by using the new log data and the trained model for inference.

The model's performance will be measured with the F1-score, which is the harmonic mean of precision and recall. This metric is useful to compare classifiers with imbalanced target labels.

### Workspace environment
For this project I used an Azure Databricks workspace.
Detailed guide on how to set-up a workspace and create compute clusters can be found [here](https://docs.microsoft.com/en-us/learn/modules/get-started-azure-databricks/)

The cluster details I used:
 - runtime version is 10.4LTS which includes `Apache Spark 3.2.1`
 - Worker: `Standard_F8` with 16GB memory, 8 cores min: `0`, max: `4` workers
 - Driver: `Standard_F16` with 32GB memory, 16 cores
 - Auto termination: `15` minutes

All of the code in the notebook uses the pyspark API and runs on any Spark cluster.

__Pricing - Be Careful! - Azure will charge you for these resources__

### The dataset
The dataset is a 12GB JSON file holding user log data and was provided through an AWS S3 bucket.
The file is not provided in this repository due to its mere size.

The full dataset I used has 26,259,199 rows and 18 columns. Each row represents a page visit by a single user. The most important features are:
 - `userId`: string, the unique id of a single user,
 - `sessionId`: long, unique id of a single session,
 - `gender`: string, gender of the user (M or F),
 - `itemInSession`: long, the cumulative number of items/pages viewed by the user in the session,
 - `length`: double, length of a song item viewed,
 - `level`: string, level of subscription (paid or free),
 - `location`: string, location of the session with city and state
 - `page`: string, page visited by the user
 - `ts`: long, time stamp UNIX epoch time in milliseconds

# Analysis
## Data cleaning
I dropped all records with no user id. These are mainly interactions before logging in or signing up to the service.
Churn as an attribute isn't explicitly defined in the dataset. Everyone who visited the 'Cancellation confirmation' page was considered to be a churning user.

## Data exploration
There are 22,277 users left in the clean dataset.
From this, 5,003 cancelled the service completely, which is 22.5% churn rate.   
![Number of days before churning](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/01_churnUserCount.jpg)

By looking at the number of days between registration and last activity, it seems that churning customers leave early on.  
Most leaving user leaves in the first three weeks.  
![Number of days before churning](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/02_NoDays.jpg)

There's no observable difference in the session starting hour proportions across the day comparing sessions when the user cancels the service against sessions with no cancellation.  
![Proportion of start hour of sessions by churning](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/03_SessionStartHour.jpg)

The figure below shows the distribution of the number of sessions before cancellation. This plot also shows that most users leave soon after sign up.  
Looking at the cumulative plot, we can see that 80% of the leaving users cancle the service within the first 20 sessions.  
![Distribution of the number of sessions before churning](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/04_NoSessions.jpg)
![Distribution of the number of sessions before churning](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/04_NoSessionsECDF.jpg)

There is no significant difference in churn rate across the genders.
![Churning and non-churning user numbers by gender](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/05_Gender.jpg)

A bar chart of the number of cancelling and non-cancelling users can be seen below.
![Number of churning and non-churning users by US State](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/06_State.jpg)

A bar plot isn't very helpful to see geographic patterns, so I created some thematic maps.
The below figure shows the number of users by state. California has a really high number of users.
![Number of users by US State](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/07_UsersMap.png)

The US of average churn rate is 22.5%, and it's fairly even across the States.
North Dakota and Montana are significantly below, whilst Soth Dakota and Maine have higher churn rates, close to 30%.
![Churn rate by US State](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/08_ChurnRate.png)

The below two plots show the user page visits. The first one shows the average page visits within a session, the second the average page visits during user lifetime.
![Churn rate by US State](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/09_pagebysession.jpg)
![Churn rate by US State](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/10_page_byuser.jpg)

## Feature extraction
for input features for modelling I selected a number of session level aggregates combined with user level aggregates.
The session Level features:
 - No of sessions
 - Average length of session
 - Average number of items in session
 - Average no of different page visits (adverts, thumbs up, thumbs down etc) in a session

User level features:
 - Days since registering
 - Total no of different page visits (adverts, thumbs up, thumbs down etc) in a session
 - Level
 - Gender
 - State

Target label:
 - churn

After processing the data and assembling the dataset ready for modelling, I split the data into two groups:
 - 80% Training set
 - 20% Test set

The below transformers were trained on the train set and applied on the test set:
 - The numeric features were centred and brought to unit variance using `StandardScaler`.
 - The gender and level features are binary categorical values, they were encoded to 0/1 values using `StringIndexer`.
 - The US State features are also categorical with 50 unique values with no ordinal relationship. First they were encoded to numeric values using `StringIndexer`, then I applied `OneHotEncoder` to turn each state (except one) into a feature of 1 or 0. I highly recommend everyone to use `handleInvalid='keep'` parameter in case you likely only have a few recordds of a given group and they may only be present in your test set. the `handleInvalid='keep'` creates an extra feature for unseen values.

All the transformed features were combined inte a `DenseVector` column using a `VectorAssembler` object instance.

The series of the above transformations of scaling, encoding and combining these into a dense vector were combined into a `Pipeline`.

## Modelling
I trained `LogisticRegression`, `RandomForestClassifier`, `GBTClassifier`, and `LinearSVC` models for this binary classification problem with their default settings. To set a baseline to compare all these (and not just to each other), I also created a dummy model predicting only 0 (not churning). Evaluating the dummy model puts the metrics of the classification models into context, particularly useful when there is a class imbalance.

The dummy classifier demonstrates 78% accuracy and a 0.68 F1 score on the training data. The performance on the test set 77% accuracy and a 0.66 F1 score.
The trained classifiers perform significantly better than the dummy classifier (considering F1 score on test data):
- `GBTClassifier` 88.2%,
- `LogisticRegression` 84.3%,
- `LinearSVC` 84.2%,
- `RandomForestClassifier` 72.3%.

![Churn rate by US State](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/11_ModelF1.jpg)

### Hyper parameter tuning

The Gradient Boosted Tree Classifier performed the best, so I chose this for further hyper-parameter tuning. During the hyper-parameter tuning, I used K-fold cross-validation with three folds. I defined the below parameters as grid points:
- Maximum depth of the tree: maxDepth=[3, 5, 8, 16]
- Fraction of the training data used for learning each decision tree: subsamplingRate: [0.7, 0.85, 1.0]  

The combination of cross-validation with three folds, four values for max depth and three for subsampling rate results in 36 individual model training and evaluation cycles. The best model has an F1 score of 87.83% whit parameters of:
 - `maxDepth=5`
 - `subsamplingRate=0.7`

This model is slightly worse than the default model. However, the difference is negligible and probably to using 3-fold cross-validation during the grid search.
Gradient Boosted Tree model provides feature importance values indicating how useful each feature is in constructing the decision trees within the model.
![Churn rate by US State](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/12_Feature%20Importance.jpg)
![Churn rate by US State](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/13_Feature%20Importance_CumSum.jpg)

The one-hot encoded US State features are at the end of the list, and their combined importance is less than 3%. US State features are likely could be removed from the model.

The top 5 most important features account for 65% of the Importance value:
- The number of days since registering: Similarly, as we saw in the exploratory analysis, most users churning will leave soon after registering.
- The number of home page visits: This feature is probably related to the above one since users logging in and starting a session will be taken to the Home page.
- The number of thumbs down: It makes sense that if people don’t like any of the songs Sparkify recommends, they will leave. We need to improve what songs are recommended as the next song.
- The number of downgrades: paid users will downgrade and then leave the service.
- The number of advertisements played: Free tier users leave because they can’t put up with the number of ads being played in-between songs.

## Recommendtaions
Beyond taking forward the model to make bespoke offers to customers likely to churn, I would like to make some recommendations to Sparkify, potential “quick wins” to consider:
- __Offer new users a one-month premium (no ads) tier subscription for no charge.__ This will take all users, including churning users behind the three weeks account age, without any ads.
- __Improve song recommendations.__ The number of thumbs down a user gives is an essential factor for predictions.

#### Conclusion

Spark is a great tool to work with big data. Most analyses on a 12GB dataset only took a few minutes to compute and visualise via the cluster I’ve been using. Training the four classification models took around an hour, and the grid search process took roughly four hours. All of the above can be executed overnight on a cloud cluster, with results ready to be looked at first thing in the morning.
With some additional budget, I believe that the grid search can be expanded to include more parameters and potentially the Logistic regression and Linear SVC models.


## Improvement
To improve the model performance it would probably worth to run an exhaustive grid search on all models. Further improvement could be made by incorporating other datasets not available from the user interaction logs.

Another important aspect is to include stakeholders from other parts of the business in the discussion at this stage. By understanding the costs associated with an additional user on the platform and the revenue yield of a user, we can refine the acceptable level of False-positive and False-negative predictions. Specifically, we can understand and estimate what it means to incentivise users who wouldn’t leave, but the modell erroneously labelled them as likely to churn. With this information, we can fine-tune the model evaluation metrics and maximise the return on investment.
