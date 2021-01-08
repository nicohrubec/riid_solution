# Riid Kaggle Solution
In this repo you can find the full code to reproduce our CV .7936 / LB .794 single LGB fold solution on Kaggle which results in a 145th rank out of 3406 teams. The task was to predict whether a student will answer a given question correctly. Metric was AUC. Competition page:
https://www.kaggle.com/c/riiid-test-answer-prediction/overview

## How to run the code:
1. Clone the repository.
2. Create a folder called 'data' and put the competition dataset in the folder. Data can be downloaded from here: https://www.kaggle.com/c/riiid-test-answer-prediction/overview
3. Install all dependencies. I included a requirements.txt file.
4. I didnt add an argparser so you will have to change the parameters in the main method. In order to do all preprocessing and train our final submission model you need to set do_merge, do_fold_split, do_val_split, do_train_lgb and full_data to True.

## Model description:
The final model uses only 35 features. Some of the most important were:
- mean answered_correctly for user
- rolling mean answered_correctly for user
- mean answered_correctly for user on question
- mean answered_correctly for user on part
- last time the user has seen the current question
- last time the user was seen
- last n times the user was seen
- user count
- user count for question
- bin questions to difficulty level and compute mean user answered_correctly per difficulty bin

Thanks to @gmilosev for the cool collaboration.
