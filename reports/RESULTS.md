# Results

Models were selected using training-only five-fold cross-validation. The fixed test set was used only after selection.

| Experiment | Selected model | Test MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Earlier grades | Previous-grade baseline | 1.00 | 2.07 | 0.79 |
| Study context only | Ridge regression | 3.50 | 4.40 | 0.06 |
| Grades + context | Previous-grade baseline | 1.00 | 2.07 | 0.79 |

MAE is the average absolute mistake in grade points. It is not a per-person confidence interval.

## Error audit

- Actual final grade = 0: n=5, MAE=6.60.
- Actual final grade > 0: n=74, MAE=0.62.

These are descriptive post-evaluation slices, not new model-selection criteria. Zero grades are retained; their causes are not inferred.

Context-only results are retrospective associations. Questionnaire measurement timing is not established, so this is not an early-warning system.
