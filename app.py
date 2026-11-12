"""Run with: streamlit run app.py. No user data is collected or uploaded."""
import json

import altair as alt
import pandas as pd
import streamlit as st
from grade_lab import ROOT, load_data, run_experiments, split_data, predict_grade, fit_studio_models
from zero_grades import SCOPE, summarize_zero_grades

st.set_page_config(page_title='Grade Lab', layout='wide')
st.caption('GRADE LAB / A SMALL DATASET, AN HONEST EXPERIMENT')
st.title('Look beyond the average.')
st.write('A simple rule won. Four difficult cases show what the average error leaves out.')


@st.cache_resource
def experiment():
    df = load_data()
    return df, run_experiments(df), fit_studio_models(df)


try:
    df, (cv, results, models, predictions), studio = experiment()
except Exception as exc:
    st.error(f'Could not load the experiment: {exc}')
    st.info('Run python grade_lab.py first to verify the bundled data and experiment.')
    st.stop()

train, test = split_data(df)
primary = results[results.experiment == 'Earlier grades'].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric('Held-out students', len(test))
c2.metric('Baseline average error', f'{primary.test_mae:.2f} / 20')
c3.metric('Error on five zero-grade cases', f'{abs(test.G3[test.G3.eq(0)] - predictions["Earlier grades"][test.G3.eq(0)]).mean():.2f} / 20')
predict, evidence, learn = st.tabs(['01 · Prediction studio', '02 · Evidence room', '03 · Learn the project'])

with predict:
    st.subheader('Watch the models disagree')
    st.write('All three approaches use the same training rows. The baseline won cross-validation; the alternatives are here to explore, not to pick a new winner from the test set.')
    left, right = st.columns([1, 1.3], gap='large')
    with left:
        selected = st.selectbox('Explore an approach', list(studio), index=list(studio).index('Previous-grade baseline'))
        if selected == 'Previous-grade baseline':
            st.info('This rule repeats G2. Moving G1 changes the nearby-example count and the other model estimates below, but not the baseline.')
        elif selected == 'Small decision tree':
            st.caption('Trees make stepwise predictions. A slider can move within a leaf without changing the estimate.')
        else:
            st.caption('Ridge combines G1 and G2. Try holding G2 fixed while moving G1.')
        g1 = st.slider('First-period grade (G1)', 0, 20, 10)
        g2 = st.slider('Second-period grade (G2)', 0, 20, 12)
        estimates = {name: predict_grade(model, g1, g2) for name, model in studio.items()}
        estimate = estimates[selected]
        st.metric('Selected estimate', f'{estimate:.1f} / 20')
        if not 0 <= estimate <= 20:
            st.warning('The raw estimate falls outside the grade range. It is not silently clipped.')
        support = ((train.G1 - g1).abs().le(2) & (train.G2 - g2).abs().le(2)).sum()
        st.caption(f'{support} training examples have both grades within ±2 points. This count is a support check, not a confidence score.')
        if support < 10:
            st.warning('Few nearby examples. Treat this scenario cautiously.')
    with right:
        vary = st.radio('Vary one input; hold the other fixed', ['G2', 'G1'], horizontal=True)
        scenario = pd.DataFrame([
            {'Input grade': value, 'Estimate': predict_grade(model, value if vary == 'G1' else g1, value if vary == 'G2' else g2), 'Approach': name}
            for name, model in studio.items() for value in range(21)])
        colors = alt.Scale(domain=list(studio), range=['#6688ff', '#dd9a47', '#25a895'])
        base_curve = alt.Chart(scenario).encode(
            x=alt.X('Input grade:Q', title=f'{vary} (other grade held fixed)', scale=alt.Scale(domain=[0, 20])),
            y=alt.Y('Estimate:Q', title='Predicted final grade'),
            color=alt.Color('Approach:N', scale=colors),
            strokeWidth=alt.condition(alt.datum.Approach == selected, alt.value(4), alt.value(1.5)),
            tooltip=['Approach', 'Input grade', alt.Tooltip('Estimate:Q', format='.2f')])
        curve = alt.layer(
            base_curve.transform_filter(alt.datum.Approach != 'Small decision tree').mark_line(),
            base_curve.transform_filter(alt.datum.Approach == 'Small decision tree').mark_line(interpolate='step-after'),
        ).properties(height=300)
        st.altair_chart(curve, width='stretch')
        st.caption('Changing a model input does not establish what would happen to a real student.')
    st.dataframe(pd.DataFrame({'Approach': list(estimates), 'Estimate / 20': list(estimates.values())}).round(2), hide_index=True, width='stretch')
    st.caption('All estimates are raw and use models fitted on the original 316 training records. The dataset is historical; this is a learning demonstration, not a student assessment tool.')

with evidence:
    st.subheader('Same split. Different information.')
    st.write('The selected model for each information set was chosen using training-only cross-validation.')
    st.dataframe(results.round(3), hide_index=True, width='stretch')
    with st.expander('See every cross-validation score'):
        st.write('Five folds on 316 training rows. Scaling is fitted inside each ridge pipeline. The test set does not select the winner.')
        st.dataframe(cv.round(3), hide_index=True)
    st.subheader('The cases behind the average')
    st.caption('Original selected grade-based approach: repeat G2. Marker area represents overlapping records; no points are jittered.')
    audit = test[['G1', 'G2', 'G3']].rename(columns={'G3': 'Actual'})
    audit['Predicted'] = predictions['Earlier grades']
    audit['Group'] = audit.Actual.eq(0).map({True: 'Zero final grade', False: 'Nonzero final grade'})
    points = audit.groupby(['Actual', 'Predicted', 'Group'], as_index=False).size().rename(columns={'size': 'Students'})
    scatter = alt.Chart(points).mark_circle(opacity=.8).encode(
        x=alt.X('Actual:Q', title='Actual final grade', scale=alt.Scale(domain=[-1, 21])),
        y=alt.Y('Predicted:Q', title='Predicted final grade', scale=alt.Scale(domain=[-1, 21])),
        color=alt.Color('Group:N', scale=alt.Scale(domain=['Nonzero final grade', 'Zero final grade'], range=['#6688ff', '#dd9a47'])),
        size=alt.Size('Students:Q', scale=alt.Scale(domain=[1, 7], range=[45, 315], zero=False), legend=alt.Legend(values=[1, 2, 4, 7])), tooltip=['Actual', 'Predicted', 'Group', 'Students'])
    diagonal = alt.Chart(pd.DataFrame({'Actual': [0, 20], 'Predicted': [0, 20]})).mark_line(strokeDash=[6, 5], color='#999999').encode(x='Actual:Q', y='Predicted:Q')
    st.altair_chart((diagonal + scatter).properties(height=360), width='stretch')
    zero = audit.Actual.eq(0)
    a, b = st.columns(2)
    a.metric('Five zero-grade cases · MAE', f'{abs(audit.loc[zero, "Actual"] - audit.loc[zero, "Predicted"]).mean():.2f}')
    b.metric('Other 74 cases · MAE', f'{abs(audit.loc[~zero, "Actual"] - audit.loc[~zero, "Predicted"]).mean():.2f}')
    st.write('One zero-grade case was predicted correctly. Four were missed; two overlap at (0, 9). The data does not establish why those students received zero.')
    st.markdown('**Held-out students with a final grade of zero**')
    st.dataframe(audit.loc[zero, ['G1', 'G2', 'Predicted', 'Actual']], hide_index=True)
    st.subheader('How much does the estimate vary?')
    report_dir = ROOT / 'reports'
    published = pd.read_csv(report_dir / 'bootstrap_intervals.csv')
    metadata = json.loads((report_dir / 'paired_comparison.json').read_text(encoding='utf-8'))
    intervals = published.loc[published.experiment.isin(['Earlier grades', 'Study context only'])]
    intervals = intervals.merge(results[['experiment', 'selected_model']], on='experiment')
    columns = {
        'experiment': 'Inputs', 'selected_model': 'Selected model', 'test_mae': 'Test MAE',
        'ci_low': '95% lower', 'ci_high': '95% upper',
    }
    st.dataframe(intervals[list(columns)].rename(columns=columns).round(2), hide_index=True)
    st.caption(
        f"Published 95% percentile intervals from {metadata['replicates']:,} paired resamples "
        f"of the {metadata['test_rows']} test records (seed {metadata['bootstrap_seed']}). "
        'They describe variation in average error with the fitted models held fixed.'
    )
    sensitivity_path = report_dir / 'split_sensitivity.csv'
    if sensitivity_path.exists():
        with st.expander('Ten additional splits: all results, no seed shopping'):
            sensitivity = pd.read_csv(sensitivity_path)
            summary = sensitivity.groupby('experiment').test_mae.agg(['mean', 'min', 'max'])
            st.dataframe(summary.round(2))
            st.caption('Added after the original analysis. Each fixed split repeats training-only selection. Splits overlap: their range is descriptive, not an independent-replication confidence interval. The original winner is unchanged.')
            st.dataframe(sensitivity.round(3), hide_index=True)
    st.subheader('Zero-grade spotlight: development records')
    st.caption(SCOPE)
    st.dataframe(summarize_zero_grades(train), hide_index=True)
    st.download_button('Download experiment results', results.to_csv(index=False), 'grade-lab-results.csv', 'text/csv')
    st.download_button('Download published bootstrap intervals', (report_dir / 'bootstrap_intervals.csv').read_bytes(),
                       'bootstrap_intervals.csv', 'text/csv')

with learn:
    st.subheader('Understand it in six moves')
    for title, body in [
        ('1. Ask a precise question', 'Can earlier grades predict a final mathematics grade in this historical sample? G2 must already be available.'),
        ('2. Inspect the data', '395 mathematics records. Grades use a 0–20 scale. Zero grades remain in the analysis.'),
        ('3. Hold out unseen examples', 'The original 80/20 split reserves 79 records. Seed 42 makes it reproducible.'),
        ('4. Compare useful baselines', 'The previous-grade rule beat ridge and the shallow tree in training cross-validation. More complexity did not help here.'),
        ('5. Inspect errors and uncertainty', 'A one-point average miss hides four large zero-grade misses. Bootstrap intervals and additional splits show different sources of variability.'),
        ('6. State what the data cannot tell you', 'Two historical schools cannot establish performance elsewhere. Context timing is unverified. Scenario curves cannot establish causal effects.')]:
        with st.expander(title):
            st.write(body)
    st.info('Do not use this demonstration for admissions, grading, student ranking, or access to support.')
    st.markdown('Data: [Paulo Cortez, Student Performance, UCI](https://doi.org/10.24432/C5TG7T) · CC BY 4.0. [Source and walkthrough](https://github.com/ran-nope/grade-lab). Built with AI assistance.')
