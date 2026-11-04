| Configuration                           | Members Count | Method   | Test Accuracy (%) | Precision (Macro) | Recall (Macro) | F1 (Macro) | ECE (%) | Latency (ms) |
| --------------------------------------- | ------------- | -------- | ----------------- | ----------------- | -------------- | ---------- | ------- | ------------ |
| Full Ensemble (All Models)              | 22            | Stacking | 96.30%            | 0.9631            | 0.9630         | 0.9630     | 0.74%   | 397.10       |
| Full Ensemble (Soft Voting)             | 22            | Soft     | 94.75%            | 0.9477            | 0.9475         | 0.9474     | 23.60%  | 571.94       |
| Full Ensemble (Hard Majority)           | 22            | Hard     | 93.65%            | 0.9366            | 0.9365         | 0.9363     | 23.60%  | 395.96       |
| Trimmed Ensemble (No Weak Members)      | 17            | Weighted | 95.70%            | 0.9571            | 0.9570         | 0.9569     | 14.06%  | 381.84       |
| Classical Supervised Only               | 10            | Weighted | 94.80%            | 0.9482            | 0.9480         | 0.9479     | 15.76%  | 186.28       |
| Deep Learning Only (CNN + MLP)          | 2             | Weighted | 96.40%            | 0.9644            | 0.9640         | 0.9640     | 2.81%   | 2.56         |
| Best Single Model (CNN (Deep Learning)) | 1             | Weighted | 97.15%            | 0.9719            | 0.9715         | 0.9715     | 1.13%   | 1.85         |
