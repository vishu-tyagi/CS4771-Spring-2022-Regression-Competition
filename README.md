# COMS4771-Spring-2022-Regression-Competition

Comptetition link - https://www.kaggle.com/competitions/coms4771-spring-2022-regression-competition/leaderboard

=========   ------
Username    vt235
Rank        1/167
=========   ------

<p align="center">
  <img src="/reports/standings.png" width="600" height="200" title="Standings">
</p>

##

Large-scale regressor for predicting trip duration for an Uber-esque transportation service.

It uses a deep neural network to regress upon features obtained from raw data as a result of feature-selection for predicting trip duration in seconds.

The neural network minimizes L1 loss and uses Adam optimizer.

The training phase includes validating the model to find the best epoch based on validation loss.
