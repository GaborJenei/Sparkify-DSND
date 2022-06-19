# Sparkify-DSND

Sparkify is a (imaginary) popular music streaming service, similar to Spotify or Pandora.
Millions of users stream music every day using either the free tier with advertisements placed in-between songs or using the monthly subscription package with no advertisements.

Users can upgrade, downgrade or cancel anytime so it's crucial they like the service

Every interaction with the service (play a song, like, login, logout, upgrade, etc.) is recorded, hoping that this dataset will be able to help us to gain some insights.

The overall goal of this project is to try to identify users before they leave so they can be offered different incentives increasing the chance of not cancelling the service.

### Contents:

#### Project Overview, Definition
Sparkify music streaming service has two subscription tiers, a paid level with no ads and free-tier with ads playing between songs. It is detrimental to Sparkify's bottom line to loose users by complete cancellation.
The primary objective of this project is to try to identify users cancelling the service.
The dataset for this is an approx 12GB log file holding all user interactions in JSON format.

#### Problem Statement
The problem which needs to be solved is clearly defined. A strategy for solving the problem, including discussion of the expected solution, has been made.

The way Sparkify generates revenue can be grouped into two main categories:
 1. Users pay a monthly fee for the Upgraded subscription tier, where users will have no advertisements
 2. Users use the free subscription tier paying no monthly fee, however they will have advertisements placed in-between songs. In this case advertising partners will pay a fee for playing their ad to our users.

If a user completely leaves Sparkify by cancelling the service, then none of the two monetisation methods will apply and total turnover of Sparkify will be less.

Based on the data provided, 22% percent of all users cancels the service, and as an approximation we can assume that this means the same proportion reduction in the turnover.

Sparkify is eager to be able to identify users likely to churn before they leave.
They will be able to offer them perks in hope of preventing them from leaving and by this increasing the revenue and profit of the company.

The expected solution is a binary classification machine learning model. This model can be used to continuously predict if a user is likely to churn or not by using the new log data and the trained model for inference.

The performance of the model is going to be measured with the F1-score, which is the harmonic mean of precision and recall. This metric is useful to compare classifiers.


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
The dataset is a 12GB JSON file holding user log data and was provided through an S3 bucket.
The file is not provided in this repository due to its mere size.

#### Analysis
##### Data exploration

##### Data cleaning
##### Feature extraction
##### Modeling

#### Conclusion
##### Reflection
Student adequately summarizes the end-to-end problem solution and discusses one or two particular aspects of the project they found interesting or difficult.
##### Improvement
Discussion is made as to how at least one aspect of the implementation could be improved. Potential solutions resulting from these improvements are considered and compared/contrasted to the current solution.
