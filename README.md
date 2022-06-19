# Sparkify-DSND

Sparkify is an imaginary popular music streaming service similar to Spotify or Pandora.
Millions of users stream music daily using either the free tier subscription with advertisements placed in-between songs or the monthly subscription package with no advertisements.
Users can upgrade, downgrade or cancel anytime, so it's crucial they like the service.
Every interaction with the service (play a song, like, login, logout, upgrade, etc.) is recorded, hoping that this dataset will help us gain some insights.
The overall goal of this project is to try to identify users before they leave so they can be offered different incentives increasing the chance of not cancelling the service.

#### Project Overview, Definition
Sparkify music streaming service has two subscription tiers, a paid level with no ads and a free tier with ads playing between songs. Therefore, it is detrimental to Sparkify's bottom line to lose users by complete cancellation.
This project's primary objective is to identify users cancelling the service.
The dataset for this is an approx 12GB log file holding all user interactions in JSON format.

#### Problem Statement
The way Sparkify generates revenue in two main ways:
 1. Users pay a monthly fee for the Upgraded subscription tier, where users will have no advertisements
 2. Users use the free subscription tier paying no monthly fee. However, they will have advertisements placed in-between songs. In this case, advertising partners will pay a fee for playing their ad to our users.

If a user leaves Sparkify by cancelling the service, then none of the two monetisation methods will apply, and the total turnover of Sparkify will be less.

Based on the data provided, 22% per cent of all users cancels the service, and as an approximation, we can assume that this means the same proportion reduction in the turnover.

Sparkify is eager to identify users likely to churn before they leave.
They will be able to offer them perks in the hope of preventing them from leaving, increasing the revenue and profit of the company.

The expected solution is a binary classification machine learning model. This model can be used to continuously predict if a user is likely to churn or not by using the new log data and the trained model for inference.

The model's performance will be measured with the F1-score, which is the harmonic mean of precision and recall. This metric is useful to compare classifiers with imbalanced target labels.

#### Workspace environment
For this project I used an Azure Databricks workspace.
Detailed guide on how to set-up a workspace and create compute clusters can be found [here](https://docs.microsoft.com/en-us/learn/modules/get-started-azure-databricks/)

The cluster details I used:
 - runtime version is 10.4LTS which includes `Apache Spark 3.2.1`
 - Worker: `Standard_F8` with 16GB memory, 8 cores min: `0`, max: `4` workers
 - Driver: `Standard_F16` with 32GB memory, 16 cores
 - Auto termination: `15` minutes

All of the code in the notebook uses the pyspark API and runs on any Spark cluster.

__Pricing - Be Careful! - Azure will charge you for these resources__

#### The dataset
The dataset is a 12GB JSON file holding user log data and was provided through an AWS S3 bucket.
The file is not provided in this repository due to its mere size.

#### Analysis
##### Data cleaning
I dropped all records with no user id. These are mainly interactions before logging in or signing up to the service.
Churn as an attribute isn't explicitly defined in the dataset. Everyone who visited the 'Cancellation confirmation' page was considered to be a churning user.

##### Data exploration
Add No churning
![alt text](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/02_NoDays.png)
add text
![alt text](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/03_SessionStartHour.png)
add text
![alt text](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/04_NoSessions.png)
add text
![alt text](https://github.com/GaborJenei/Sparkify-DSND/blob/main/eda_visuals/04_NoSessionsECDF.png)


##### Feature extraction
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


##### Modelling

#### Conclusion

##### Reflection
Student adequately summarizes the end-to-end problem solution and discusses one or two particular aspects of the project they found interesting or difficult.
##### Improvement
Discussion is made as to how at least one aspect of the implementation could be improved. Potential solutions resulting from these improvements are considered and compared/contrasted to the current solution.
