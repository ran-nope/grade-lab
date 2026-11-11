"""Run with: streamlit run app.py"""
import pandas as pd
import streamlit as st
from grade_lab import load_data, run_experiments, split_data, predict_grade

st.set_page_config(page_title='Grade Lab', page_icon='🧪', layout='wide')
st.markdown('''<style>
.stApp {background: #f5f7fb;}
h1,h2,h3 {color: #182442;}
div[data-testid="stMetric"] {background:white;padding:20px;border-radius:16px;border:1px solid #e3e8f2;}
</style>''', unsafe_allow_html=True)
st.caption('AN OPEN MACHINE LEARNING LAB · UCI MATHEMATICS DATA')
st.title('Grades are signals. Not destinies.')
st.write('Explore what a small model can learn—and where its confidence runs out.')


@st.cache_resource
def experiment():
    df = load_data()
    return df, run_experiments(df)


try:
    df, (cv, results, models, predictions) = experiment()
except Exception as exc:
    st.error(f'Could not load the experiment: {exc}')
    st.info('Run python grade_lab.py first to download the public data and verify the experiment.')
    st.stop()

train, test = split_data(df)
primary = results[results.experiment == 'Earlier grades'].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric('Students in the dataset', len(df))
c2.metric('Students held out for testing', len(test))
c3.metric('Average test error', f'{primary.test_mae:.2f} / 20')
predict, evidence, learn = st.tabs(['01 · Prediction studio', '02 · Evidence room', '03 · Learn the project'])

with predict:
    left, right = st.columns([1, 1.3], gap='large')
    with left:
        st.subheader('Try an example')
        st.write('This model uses two earlier grades, after the second grading period.')
        g1 = st.slider('First-period grade (G1)', 0, 20, 10)
        g2 = st.slider('Second-period grade (G2)', 0, 20, 12)
        estimate = predict_grade(models['Earlier grades'], g1, g2)
        st.metric('Estimated final grade', f'{estimate:.1f} / 20')
        if not 0 <= estimate <= 20:
            st.warning('This raw model estimate is outside the valid grade range: an example of model limitations.')
        support = ((train.G1 - g1).abs().le(2) & (train.G2 - g2).abs().le(2)).sum()
        st.caption(f'{support} training examples have both earlier grades within ±2 points. This is a support check, not a confidence score.')
        if support < 10:
            st.warning('Few nearby examples: treat this scenario with extra caution.')
    with right:
        st.subheader('What if the second grade changed?')
        scenario = pd.DataFrame({'Second-period grade': range(21), 'Model estimate': [predict_grade(models['Earlier grades'], g1, x) for x in range(21)]})
        st.line_chart(scenario.set_index('Second-period grade'), color='#4361ee')
        st.info('This shows how the model responds to different inputs. It does not prove that changing an input causes the outcome to change.')
        st.caption(f'Selected approach: {primary.selected_model}. The app uses the same training-only fitted model evaluated in the report.')
        if primary.selected_model == 'Previous-grade baseline':
            st.caption('The winning rule returns G2 unchanged. G1 affects the nearby-example count, but not this rule’s estimate. The trained models did not beat it in cross-validation.')
    st.warning('Learning demonstration only. This historical, two-school sample cannot determine a student’s ability, admissions, support eligibility, or future.')

with evidence:
    st.subheader('The information experiment')
    st.write('Same students. Same test split. Three sets of inputs. How much do earlier grades change prediction error?')
    st.bar_chart(results.set_index('experiment')[['test_mae']], horizontal=True, color='#4361ee')
    st.dataframe(results.round(3), hide_index=True, use_container_width=True)
    st.caption('Lower MAE is better. Context means study time, past failures, and travel time; its measurement timing is not verified.')
    with st.expander('How were models selected?'):
        st.write('Five-fold cross-validation on 316 training records. Mean, ridge regression, and a shallow tree compete; grade-based experiments also include the strong “just use G2” baseline. No test scores determine the winner.')
        st.dataframe(cv.round(3), hide_index=True)
    st.subheader('Where does it go wrong?')
    audit = pd.DataFrame({'Actual': test.G3, 'Predicted': predictions['Earlier grades']})
    audit['Absolute error'] = (audit.Actual - audit.Predicted).abs()
    st.scatter_chart(audit, x='Actual', y='Predicted', color='#139c90')
    for label, mask in [('Zero final grades', test.G3 == 0), ('Nonzero final grades', test.G3 > 0)]:
        group = audit.loc[mask]
        st.write(f'**{label}:** {len(group)} test records · MAE {group["Absolute error"].mean():.2f} points')
    st.caption('Zeros are retained. The data does not establish why these students received zero. These slices describe errors after evaluation.')
    st.download_button('Download experiment results', results.to_csv(index=False), 'grade-lab-results.csv', 'text/csv')

with learn:
    st.subheader('Understand it in six moves')
    for title, body in [
        ('1. Ask a precise question', 'Can first- and second-period grades predict a final mathematics grade in this historical sample?'),
        ('2. Inspect the data', '395 mathematics records. G1, G2, and G3 use a 0–20 scale. Keep zero grades; do not silently clean away hard examples.'),
        ('3. Hold out unseen examples', 'Use 80% for training and 20% for testing. Fix the random seed so other people can reproduce the split.'),
        ('4. Beat useful baselines', 'Compare with predicting the average and with simply repeating G2. A complex model has to earn its complexity.'),
        ('5. Measure honest errors', 'MAE means average absolute error in grade points. Cross-validation chooses models without peeking at test results.'),
        ('6. Explain the limits', 'Two schools are not the world. Correlation is not causation. A model estimate is not a verdict about a person.')]:
        with st.expander(title):
            st.write(body)
    st.markdown('Data: [Paulo Cortez, Student Performance, UCI](https://doi.org/10.24432/C5TG7T) · CC BY 4.0. Built with AI assistance as a learning project.')
