Tutorials!
==========

More tutorials are on the way! In the meantime, check out the EEG2025 Competition lessons and our EEGDash basics guide.


# Possibles approaches to define the windows could be:
#     _____________|_______________|_________________|____________
#                stimulus.         response.       feedback.
# Op1: *************
# Op2:                    ******************* (Using + 2.0)
# Op3:                 ******************** (Using + 0.5)
# Op4:               *****************
####################################################################

option 1: no ERP
Baseline RMSE (predict train mean): 0.3309
Using n_filters=6 (n_channels=128) with 200 time points
Fixed SPoC (nfilter=6) + Ridge(alpha=1.0)
Train RMSE: 0.3601 | R^2: 0.0389
Valid RMSE: 0.3403 | R^2: -0.2015
Test RMSE: 0.3669 | R^2: -0.3361

option 2: ERP
Baseline RMSE (predict train mean): 0.3309
Using n_filters=6 (n_channels=128) with 200 time points
Fixed SPoC (nfilter=6) + Ridge(alpha=1.0)
Train RMSE: 0.3634 | R^2: 0.0212
Valid RMSE: 0.3363 | R^2: -0.1736
Test RMSE: 0.3492 | R^2: -0.2105

option 3: ERP
Using n_filters=6 (n_channels=128) with 200 time points
Fixed SPoC (nfilter=6) + Ridge(alpha=1.0)
Train RMSE: 0.3607 | R^2: 0.0357
Valid RMSE: 0.3457 | R^2: -0.2400
Test  RMSE: 0.3306 | R^2: -0.0846

option 4:
Baseline RMSE (predict train mean): 0.3309
Using n_filters=6 (n_channels=128) with 200 time points
Fixed SPoC (nfilter=6) + Ridge(alpha=1.0)
Train RMSE: 0.3631 | R^2: 0.0228
Valid RMSE: 0.3401 | R^2: -0.1998
Test  RMSE: 0.3463 | R^2: -0.1906