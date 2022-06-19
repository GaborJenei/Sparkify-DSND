# Sparkify-DSND

Sparkify is an imaginary popular music streaming service similar to Spotify or Pandora.
Millions of users stream music daily using either the free tier subscription with advertisements placed in-between songs or the monthly subscription package with no advertisements.
Users can upgrade, downgrade or cancel anytime, so it's crucial they like the service.
Every interaction with the service (play a song, like, login, logout, upgrade, etc.) is recorded, hoping that this dataset will help us gain some insights.
The overall goal of this project is to try to identify users before they leave so they can be offered different incentives increasing the chance of not cancelling the service.

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
 - itemInSession: long, the cumulative number of items/pages viewed by the user in the session,
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
Add No churning

By looking at the number of days between registration and last activity, it seems that churning customers leave early on.
Most leaving user leaves in the first three weeks.
![Number of days before churning](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/02_NoDays.png)

There's no observable difference in the session starting hour proportions across the day comparing sessions when the user cancels the service against sessions with no cancellation.
![Proportion of start hour of sessions by churning](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/03_SessionStartHour.png)

The figure below shows the distribution of the number of sessions before cancellation. This plot also shows that most users leave soon after sign up.
Looking at the cumulative plot, we can see that 80% of the leaving users cancle the service within the first 20 sessions.
![Distribution of the number of sessions before churning](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/04_NoSessions.png)
![Distribution of the number of sessions before churning](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/04_NoSessionsECDF.png)

There is no significant difference in churn rate across the genders.
![Churning and non-churning user numbers by gender](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/05_Gender.png)

A bar chart of the number of cancelling and non-cancelling users can be seen below.
![Number of churning and non-churning users by US State](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/06_State.png)

A bar plot isn't very helpful to see geographic patterns, so I created some thematic maps.
The below figure shows the number of users by state. California has a really high number of users.
![Number of users by US State](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/07_UsersMap.png)

The US of average churn rate is 22.5%, and it's fairly even across the States.
North Dakota and Montana are significantly below, whilst Soth Dakota and Maine have higher churn rates, close to 30%.
![Churn rate by US State](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/08_ChurnRate.png)

The below two plots show the user page visits. The first one shows the average page visits within a session, the second the average page visits during user lifetime.
![Churn rate by US State](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/09_pagebysession.png)
![Churn rate by US State](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/10_page_byuser.png)

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

### Hyper parameter tuning

#### Conclusion

## Reflection
Student adequately summarizes the end-to-end problem solution and discusses one or two particular aspects of the project they found interesting or difficult.
Spark is a great tool to work with big data.

## Improvement
To improve the model performance it would probably worth to run an exhaustive grid search on all models. Further improvement could be made by incorporating other datasets not available from the user interaction logs. It would be also an interesting discussion with business stakeholders what incentives are they proposing to offer to customers leaving, what is the cost of that against the revenue a single customer yields. This can help to establish what the Return on Investment is and help further refine the models specificity and sensitivity. I.e., what are the implications of falsely labelling a customer as churning and not identifying customers leaving.
