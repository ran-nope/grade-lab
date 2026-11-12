"""Run with: streamlit run app.py. No user data is collected or uploaded."""
import json

import altair as alt
import pandas as pd
import streamlit as st
from grade_lab import ROOT, load_data, run_experiments, run_expanded_benchmarks, split_data, predict_grade, fit_studio_models
from robustness import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, bootstrap_mae, conformal_bounds, studio_calibration
from zero_grades import SCOPE, summarize_zero_grades

st.set_page_config(page_title='Grade Lab', layout='wide')
st.caption('GRADE LAB / A SMALL DATASET, AN HONEST EXPERIMENT')
st.title('Look beyond the average.')
cohort = st.sidebar.selectbox('Subject', ['Mathematics', 'Portuguese'])
subject = 'mat' if cohort == 'Mathematics' else 'por'


@st.cache_resource
def experiment(subject):
    df = load_data(subject)
    train, test = split_data(df)
    original = run_experiments(df)
    calibrated, coverage = studio_calibration(train, test)
    return train, test, original, fit_studio_models(df), calibrated, coverage, run_expanded_benchmarks(df)


try:
    train, test, (cv, results, _, predictions), studio, calibrated, coverage, expanded = experiment(subject)
except Exception as exc:
    st.error(f'Could not load the experiment: {exc}')
    st.info('Run python grade_lab.py first to verify the bundled data and experiment.')
    st.stop()

st.write(f'{cohort}: {len(train) + len(test)} records. Compare model predictions and inspect their errors.')
primary = results[results.experiment == 'Earlier grades'].iloc[0]
zero_test = test.G3.eq(0)
audit_mae = abs(test.G3.to_numpy() - predictions['Earlier grades'])
c1, c2, c3 = st.columns(3)
c1.metric('Held-out students', len(test))
c2.metric('Earlier-grades average error', f'{primary.test_mae:.2f} / 20')
c3.metric(f'Error on {zero_test.sum()} zero-grade cases',
          f'{audit_mae[zero_test].mean():.2f} / 20' if zero_test.any() else 'Unavailable')
predict, evidence, learn = st.tabs(['01 · Prediction studio', '02 · Evidence room', '03 · Learn the project'])

with predict:
    st.subheader('Watch the models disagree')
    st.write(f'All three approaches use the same development rows. The earlier-grades experiment selected {primary.selected_model.lower()} in cross-validation.')
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
        inputs = pd.DataFrame({'G1': g1, 'G2': g2}, index=range(21))
        inputs[vary] = inputs.index
        scenario = pd.concat([
            pd.DataFrame({'Input grade': inputs.index, 'Estimate': model.predict(inputs), 'Approach': name})
            for name, model in studio.items()
        ], ignore_index=True)
        colors = alt.Scale(domain=list(studio), range=['#6688ff', '#dd9a47', '#25a895'])
        base_curve = alt.Chart(scenario).encode(
            x=alt.X('Input grade:Q', title=f'{vary} (other grade held fixed)', scale=alt.Scale(domain=[0, 20])),
            y=alt.Y('Estimate:Q', title='Predicted final grade'),
            color=alt.Color('Approach:N', scale=colors),
            strokeWidth=alt.condition(alt.datum.Approach == selected, alt.value(4), alt.value(1.5)),
            tooltip=['Approach', 'Input grade', alt.Tooltip('Estimate:Q', format='.2f')])
        fitted, q_hat = calibrated[selected]
        center = fitted.predict(inputs)
        lower, upper = conformal_bounds(center, q_hat)
        band_data = pd.DataFrame({'Input grade': inputs.index, 'Center': center, 'Lower': lower, 'Upper': upper})
        interpolation = 'step-after' if selected == 'Small decision tree' else 'linear'
        band_base = alt.Chart(band_data).encode(
            x=alt.X('Input grade:Q', title=f'{vary} (other grade held fixed)', scale=alt.Scale(domain=[0, 20])),
        )
        band = band_base.mark_area(opacity=0.2, color='#6688ff', interpolate=interpolation).encode(
            y=alt.Y('Lower:Q', title='Predicted final grade'), y2='Upper:Q',
            tooltip=['Input grade:Q', alt.Tooltip('Lower:Q', format='.2f'), alt.Tooltip('Upper:Q', format='.2f')],
        )
        center_line = band_base.mark_line(strokeDash=[5, 4], color='#6688ff', interpolate=interpolation).encode(
            y=alt.Y('Center:Q', title='Predicted final grade'),
        )
        curve = alt.layer(
            band,
            base_curve.transform_filter(alt.datum.Approach != 'Small decision tree').mark_line(),
            base_curve.transform_filter(alt.datum.Approach == 'Small decision tree').mark_line(interpolate='step-after'),
            center_line,
        ).properties(height=300)
        st.altair_chart(curve, width='stretch')
        st.caption(f'Shading shows the nominal 90% conformal interval for {selected.lower()}. '
                   'The dashed center uses the fit trained before calibration. The solid curves use all development rows.')
        st.caption('Changing a model input does not establish what would happen to a real student.')
    st.dataframe(pd.DataFrame({'Approach': list(estimates), 'Estimate / 20': list(estimates.values())}).round(2), hide_index=True, width='stretch')
    st.caption(f'The estimate table uses models fitted on all {len(train)} development records.')

with evidence:
    st.subheader('Same split. Different information.')
    st.write('The selected model for each information set was chosen using training-only cross-validation.')
    st.dataframe(results.round(3), hide_index=True, width='stretch')
    with st.expander('See every cross-validation score'):
        st.write(f'Five folds on {len(train)} development rows. Each ridge pipeline fits scaling within its training fold.')
        st.dataframe(cv.round(3), hide_index=True)
    st.subheader('The cases behind the average')
    st.caption(f'Selected earlier-grades model: {primary.selected_model}. Marker area shows the number of overlapping records.')
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
    a.metric(f'{zero.sum()} zero-grade cases · MAE', f'{audit_mae[zero].mean():.2f}' if zero.any() else 'Unavailable')
    b.metric(f'{(~zero).sum()} other cases · MAE', f'{audit_mae[~zero].mean():.2f}' if (~zero).any() else 'Unavailable')
    st.write(f'{(zero & audit.Predicted.eq(0)).sum()} of the {zero.sum()} zero-grade cases were predicted exactly.')
    st.markdown('**Held-out students with a final grade of zero**')
    st.dataframe(audit.loc[zero, ['G1', 'G2', 'Predicted', 'Actual']], hide_index=True)
    st.subheader('How much does the estimate vary?')
    report_dir = ROOT / 'reports'
    if subject == 'mat':
        published = pd.read_csv(report_dir / 'bootstrap_intervals.csv')
        metadata = json.loads((report_dir / 'paired_comparison.json').read_text(encoding='utf-8'))
        bootstrap_download = (report_dir / 'bootstrap_intervals.csv').read_bytes()
    else:
        published, _ = bootstrap_mae(test.G3, predictions)
        metadata = {'replicates': BOOTSTRAP_REPLICATES, 'test_rows': len(test), 'bootstrap_seed': BOOTSTRAP_SEED}
        bootstrap_download = published.to_csv(index=False)
    intervals = published.loc[published.experiment.isin(['Earlier grades', 'Study context only'])]
    intervals = intervals.merge(results[['experiment', 'selected_model']], on='experiment')
    columns = {
        'experiment': 'Inputs', 'selected_model': 'Selected model', 'test_mae': 'Test MAE',
        'ci_low': '95% lower', 'ci_high': '95% upper',
    }
    st.dataframe(intervals[list(columns)].rename(columns=columns).round(2), hide_index=True)
    st.caption(
        f"95% percentile intervals from {metadata['replicates']:,} paired resamples "
        f"of the {metadata['test_rows']} test records (seed {metadata['bootstrap_seed']}). "
        'They describe variation in average error with the fitted models held fixed.'
    )
    st.subheader('Conformal coverage on held-out records')
    st.dataframe(coverage.rename(columns={'test_coverage': 'Empirical test coverage'}).round(3), hide_index=True)
    st.caption('The fixed studio models use earlier grades. Each fit sets aside 25% of development records for calibration. '
               'Coverage is the fraction of held-out grades within the nominal 90% intervals.')
    st.subheader('Exploratory expanded-context benchmarks')
    st.dataframe(expanded.round(3), hide_index=True, width='stretch')
    st.caption('These comparisons use historical records. The timing of the absences field is unverified. '
               'The models and hurdle threshold were fixed before evaluation.')
    sensitivity_path = report_dir / 'split_sensitivity.csv'
    if subject == 'mat' and sensitivity_path.exists():
        with st.expander('Ten additional splits: all results, no seed shopping'):
            sensitivity = pd.read_csv(sensitivity_path)
            summary = sensitivity.groupby('experiment').test_mae.agg(['mean', 'min', 'max'])
            st.dataframe(summary.round(2))
            st.caption('Added after the original analysis. Each fixed split repeats training-only selection. Splits overlap: their range is descriptive, not an independent-replication confidence interval. The original winner is unchanged.')
            st.dataframe(sensitivity.round(3), hide_index=True)
    st.subheader('Zero-grade spotlight: development records')
    st.caption(SCOPE if subject == 'mat' else
               f'This describes the development data. The comparison cannot establish causes. '
               f'The {zero.sum()} held-out zero-grade cases were excluded.')
    st.dataframe(summarize_zero_grades(train), hide_index=True)
    st.download_button('Download experiment results', results.to_csv(index=False), 'grade-lab-results.csv', 'text/csv')
    st.download_button('Download bootstrap intervals', bootstrap_download,
                       'bootstrap_intervals.csv', 'text/csv')

with learn:
    st.subheader('Understand it in six moves')
    for title, body in [
        ('1. Ask a precise question', f'Can earlier grades predict a final {cohort.lower()} grade in this sample? G2 must already be available.'),
        ('2. Inspect the data', f'{len(train) + len(test)} records on a 0 to 20 grade scale, including zero grades.'),
        ('3. Hold out unseen examples', f'The 80/20 split reserves {len(test)} records with seed 42.'),
        ('4. Compare useful baselines', 'Compare the previous-grade rule, ridge, and the shallow tree using cross-validation on development records.'),
        ('5. Inspect errors and uncertainty', 'Compare errors by final grade group. Bootstrap and conformal intervals describe different quantities.'),
        ('6. State what the data cannot tell you', 'Two historical schools cannot establish performance elsewhere. Context timing is unverified. Scenario curves cannot establish causal effects.')]:
        with st.expander(title):
            st.write(body)
    st.info('Do not use this demonstration for admissions, grading, student ranking, or access to support.')
    st.markdown('Data: [Paulo Cortez, Student Performance, UCI](https://doi.org/10.24432/C5TG7T) · CC BY 4.0. [Source and walkthrough](https://github.com/ran-nope/grade-lab). Built with AI assistance.')
