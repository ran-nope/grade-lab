# Your first ML project: a guided tour

## First 20 minutes

Run the app. Move the two sliders. Find a scenario with few nearby examples. Open the Evidence room and locate the selected grade-only model. Ask: did it beat simply repeating the previous grade?

## Read the notebook next

Read one section, run its cell, then explain the output aloud. You should be able to explain what one row represents, which column is the target, and why not every available column belongs in the input.

## Vocabulary in plain English

| Term | Meaning here |
|---|---|
| Feature | An input such as the second-period grade |
| Target | G3, the final grade we want to predict |
| Training | Finding patterns using the development records |
| Test set | Records withheld to check generalization |
| Baseline | A simple approach your model should compete against |
| MAE | Average absolute difference between prediction and reality |
| Cross-validation | Repeatedly hiding part of training data to compare models |
| Leakage | Letting information unavailable at prediction or evaluation time influence learning |
| Overfitting | Learning peculiarities that do not carry over to unseen examples |
| Ridge regression | A linear model with a penalty that discourages large coefficients |

## Work through the code

1. `load_data`: read the semicolon-separated file and check values.
2. `split_data`: separate training from testing.
3. `candidates`: see the simple competing approaches.
4. `run_experiments`: select using training CV, evaluate afterward.
5. `save_report`: turn actual outputs into evidence someone else can inspect.
6. `app.py`: connect the model and analysis to an interactive interface.

## Explain this project honestly in an interview

“I built this learning project with AI assistance. It compares grade-based and context-based prediction using a public student dataset. I can reproduce the results and explain the train/test split, baseline comparison, and leakage checks. Its historical sample and unverified questionnaire timing limit what we can claim.”

Only claim understanding after you can answer the questions below. Replace this description with your own words and describe any changes you personally make.

## Questions worth answering

- Why is predicting G2 a stronger baseline than predicting the average?
- What does an MAE of 2 mean? Why does it not mean “every estimate is within 2 points”?
- Why is scaling fitted separately inside each validation fold?
- Why would adding the final grade as an input invalidate the experiment?
- Why does changing study time in a model not prove that studying more causes a specific improvement?
- Why might zero-grade cases be harder, and what can't we infer about them?

## First independent change

Add `DummyRegressor(strategy='median')` to `candidates`. Predict how it will compare with the mean baseline, then rerun the experiment. Record your expectation, the actual result, and your explanation. Treat the existing held-out set as previously inspected; use fresh validation for any later performance claims.
