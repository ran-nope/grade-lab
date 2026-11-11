"""Rebuild the banner and social image from actual seed-42 held-out predictions."""
from pathlib import Path
from math import sqrt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / 'reports/held_out_predictions.csv')
points = data.groupby(['actual', 'predicted'], as_index=False).size()
svg = ['''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 360" role="img" aria-labelledby="title desc">
<title id="title">Grade Lab: look beyond the average</title>
<desc id="desc">Actual held-out predictions from the repeat-G2 baseline. Five zero-grade cases are orange: one correct at zero, and four misses, including two overlapping at predicted nine. Marker area shows record count.</desc>
<defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#182442"/><stop offset="1" stop-color="#0f1730"/></linearGradient></defs>
<rect width="1280" height="360" fill="url(#bg)"/>
<g font-family="'Segoe UI', Helvetica, Arial, sans-serif">
<text x="70" y="100" font-size="15" letter-spacing="3" fill="#a4b6e6">AN OPEN MACHINE LEARNING LAB</text>
<text x="66" y="183" font-size="78" font-weight="700" fill="white">Grade Lab</text>
<text x="70" y="230" font-size="26" fill="#d8e0f5">Look beyond the average.</text>
<text x="70" y="274" font-size="17" fill="#f4a261">1.00 average error. Four large zero-grade misses.</text>
<text x="70" y="311" font-size="14" fill="#a4b6e6">Real predictions · Model comparison · Uncertainty checks</text>
''']
for grade in [0, 5, 10, 15, 20]:
    x, y = 840 + 17.5 * grade, 290 - 11 * grade
    svg.append(f'<line x1="840" y1="{y}" x2="1190" y2="{y}" stroke="#2b3a64" stroke-dasharray="2 6"/>')
    svg.append(f'<text x="{x}" y="310" text-anchor="middle" fill="#a4b6e6" font-size="12">{grade}</text>')
    svg.append(f'<text x="821" y="{y+4}" text-anchor="end" fill="#a4b6e6" font-size="12">{grade}</text>')
svg.append('<line x1="840" y1="290" x2="1190" y2="70" stroke="#a4b6e6" stroke-dasharray="6 5"/>')
for row in points.itertuples():
    x, y = 840 + 17.5 * row.actual, 290 - 11 * row.predicted
    color = '#f4a261' if row.actual == 0 else '#8199ff'
    r = 4 * sqrt(row.size)
    svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}" fill="{color}" opacity="0.78"><title>Actual {row.actual}, predicted {row.predicted:g}: {row.size} students</title></circle>')
    if row.actual == 0:
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r+4:.2f}" fill="none" stroke="{color}" stroke-width="1.5"/>')
svg += ['''<text x="1015" y="337" text-anchor="middle" font-size="13" fill="#a4b6e6">actual final grade</text>
<text x="786" y="180" text-anchor="middle" font-size="13" fill="#a4b6e6" transform="rotate(-90 786 180)">predicted grade</text>
<text x="840" y="38" font-size="13" fill="#f4a261">Orange: zero final grades (5 records)</text>
<text x="840" y="55" font-size="11" fill="#a4b6e6">79 test records · overlapping records grouped</text>
</g></svg>''']
(ROOT / 'assets/banner.svg').write_text('\n'.join(svg), encoding='utf-8')
fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor='#182442')
fig.text(.06, .82, 'GRADE LAB', color='white', fontsize=36, weight='bold')
fig.text(.06, .64, 'Look beyond\nthe average.', color='#d8e0f5', fontsize=30, linespacing=1.2)
fig.text(.06, .40, '1.00 point average error\n6.60 on five zero-grade cases', color='#f4a261', fontsize=17, linespacing=1.5)
fig.text(.06, .18, 'Model comparison · Real error analysis\nReproducible checks · Interactive app', color='#a4b6e6', fontsize=14, linespacing=1.5)
ax = fig.add_axes([.56, .18, .39, .65], facecolor='#182442')
for is_zero, group in points.groupby(points.actual.eq(0)):
    ax.scatter(group.actual, group.predicted, s=group['size']*28, color='#f4a261' if is_zero else '#8199ff', alpha=.85)
ax.plot([0,20],[0,20],'--',color='#a4b6e6',linewidth=1)
ax.set(xlim=(-1,21),ylim=(-1,21),xlabel='Actual final grade',ylabel='Predicted grade',title='79 held-out records · baseline predictions')
ax.tick_params(colors='#a4b6e6')
ax.xaxis.label.set_color('#d8e0f5'); ax.yaxis.label.set_color('#d8e0f5'); ax.title.set_color('#d8e0f5')
for spine in ax.spines.values(): spine.set_color('#526588')
fig.savefig(ROOT / 'assets/social-preview.png', facecolor=fig.get_facecolor())
plt.close(fig)
print('Banner and social image rebuilt from actual predictions.')

