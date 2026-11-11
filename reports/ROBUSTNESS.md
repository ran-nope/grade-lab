# How stable are the results?

## Fixed-test bootstrap

95% percentile intervals from 10,000 paired resamples of the 79 seed-42 test rows (bootstrap seed 2026). Models are held fixed. These intervals describe test-row sampling variability under an IID assumption; they omit retraining, model-selection, school clustering, and distribution-shift uncertainty. They are not intervals for individual students.

| Experiment | Test MAE | 95% bootstrap interval |
|---|---:|---:|
| Earlier grades | 1.00 | 0.63–1.43 |
| Study context only | 3.50 | 2.93–4.09 |
| Grades + context | 1.00 | 0.63–1.43 |

The paired context-minus-grades MAE difference is 2.50 points (95% interval 2.03–2.97). This compares different information sets, not ridge against the baseline with identical inputs.

## Additional split sensitivity

This supplementary analysis was added after inspecting the original seed-42 results. All ten seeds were fixed before this analysis ran: 7, 17, 27, 37, 47, 57, 67, 77, 87, 97. Each split reruns training-only five-fold selection, then evaluates its selected model on that split’s held-out rows. No seed is selected as the best result.

The splits overlap, so their scores are correlated. The range below is descriptive, not a confidence interval or ten independent replications. This is a sensitivity analysis, not new untouched external validation.

| Experiment | Mean MAE across splits | Min–max MAE |
|---|---:|---:|
| Earlier grades | 0.89 | 0.66–1.23 |
| Study context only | 3.24 | 2.97–3.53 |
| Grades + context | 0.89 | 0.66–1.23 |

All selected models, seeds, zero-grade counts, and scores are in [split_sensitivity.csv](split_sensitivity.csv). None of these checks changes the original winner or app defaults. With only five zero-grade cases in the original test set, that subgroup’s error remains especially uncertain.
