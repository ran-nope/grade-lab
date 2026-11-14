# Cross-school holdout

All models receive G1 and G2. Fixed settings match the original experiment. Scaling is fitted only on the training school; no evaluation-school labels select models.

| direction | model | train_n | test_n | mae |
|---|---|---|---|---|
| GP → MS | Ridge regression | 349 | 46 | 1.091 |
| GP → MS | Small decision tree | 349 | 46 | 1.305 |
| GP → MS | Previous-grade baseline | 349 | 46 | 0.913 |
| MS → GP | Ridge regression | 46 | 349 | 1.319 |
| MS → GP | Small decision tree | 46 | 349 | 2.541 |
| MS → GP | Previous-grade baseline | 46 | 349 | 0.968 |

MS has only 46 students versus 349 at GP. Training on MS is particularly unstable; these two historical schools do not establish generalization to other schools or years. Both directions are descriptive stress tests, not a new model-selection procedure.
