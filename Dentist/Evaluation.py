import json
import re
import numpy as np
import pandas as pd

with open('results.json') as f:
    results = json.load(f)

print(f'Total results: {len(results)}')
print(f'Models: {set(r["model"] for r in results)}')
print(f'Question types: {set(r["type"] for r in results)}')
print(f'Sample result keys: {list(results[0].keys())}')


# ── Scoring functions ─────────────────────────────────────────────────────────

def extract_number(text):
    """Extract first number from a string, handling negatives and decimals."""
    if text in [None, 'ERROR', '']:
        return None
    matches = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', str(text))
    if matches:
        return float(matches[0])
    return None


def score_numerical(predicted, ground_truth, tolerance=0.20):
    """
    Score numerical answers using relative error.
    Returns 1.0 if within tolerance, scales down otherwise.
    tolerance=0.20 means within 20% counts as correct.
    """
    pred = extract_number(predicted)
    gt = extract_number(str(ground_truth))

    if pred is None or gt is None:
        return 0.0

    if gt == 0:
        return 1.0 if abs(pred) < 1e-6 else 0.0

    relative_error = abs(pred - gt) / abs(gt)

    return 1.0 if relative_error <= tolerance else 0.0

def score_with_multiple_tolerances(predicted, ground_truth, tolerances=[0.20, 0.30, 0.50]):
    """
    Score the same prediction at multiple tolerance levels.
    Returns a dict of {tolerance: score}.
    """
    return {
        f"tol_{int(t*100)}": score_numerical(predicted, ground_truth, tolerance=t)
        for t in tolerances
    }

def score_exact(predicted, ground_truth):
    """Exact match for categorical answers (nbars, ngaussians)."""
    pred = extract_number(predicted)
    gt = extract_number(str(ground_truth))
    if pred is None or gt is None:
        return 0.0
    return 1.0 if int(round(pred)) == int(round(gt)) else 0.0


def score_answer(predicted, ground_truth, qtype):
    if qtype in ['nbars', 'ngaussians']:
        return score_exact(predicted, ground_truth)
    else:
        return score_numerical(predicted, ground_truth)


print('Scoring functions defined.')


# ── Compute scores for all results ───────────────────────────────────────────

for r in results:
    r['baseline_score'] = score_answer(
        r['baseline_answer'], r['ground_truth'], r['type']
    )
    r['revised_score'] = score_answer(
        r['revised_answer'], r['ground_truth'], r['type']
    )

    if r['revised_score'] > r['baseline_score']:
        r['dentist_effect'] = 'improved'
    elif r['revised_score'] < r['baseline_score']:
        r['dentist_effect'] = 'worsened'
    else:
        r['dentist_effect'] = 'unchanged'

df = pd.DataFrame(results)
print(f'Scored {len(df)} results')
print(df[['model','type','baseline_score','revised_score','dentist_effect']].head(10).to_string())


# ── Main results table ───────────────────────────────────────────────────────

print('=' * 70)
print('MAIN RESULTS: Accuracy by Model and Question Type')
print('=' * 70)

for model in ['Qwen2VL', 'LLaVA']:
    print(f'\n--- {model} ---')
    model_df = df[df['model'] == model]

    print(f'  {"Question Type":<15} {"Baseline":>10} {"Revised":>10} {"Change":>10} {"N":>5}')
    print(f'  {"-"*15} {"-"*10} {"-"*10} {"-"*10} {"-"*5}')

    for qtype in sorted(df['type'].unique()):
        qdf = model_df[model_df['type'] == qtype]
        if len(qdf) == 0:
            continue
        b = qdf['baseline_score'].mean()
        r = qdf['revised_score'].mean()
        change = r - b
        marker = '↑' if change > 0.01 else ('↓' if change < -0.01 else '=')
        print(f'  {qtype:<15} {b:>10.3f} {r:>10.3f} {change:>+9.3f}{marker} {len(qdf):>5}')

    b_all = model_df['baseline_score'].mean()
    r_all = model_df['revised_score'].mean()
    change_all = r_all - b_all
    marker = '↑' if change_all > 0.01 else ('↓' if change_all < -0.01 else '=')
    print(f'  {"OVERALL":<15} {b_all:>10.3f} {r_all:>10.3f} {change_all:>+9.3f}{marker} {len(model_df):>5}')


# ── Dentist effect breakdown ─────────────────────────────────────────────────

print('\n' + '=' * 70)
print('DENTIST EFFECT')
print('=' * 70)

for model in ['Qwen2VL', 'LLaVA']:
    model_df = df[df['model'] == model]
    counts = model_df['dentist_effect'].value_counts()
    total = len(model_df)

    print(f'\n{model} (n={total}):')
    for effect in ['improved', 'unchanged', 'worsened']:
        n = counts.get(effect, 0)
        pct = n / total * 100
        print(f'  {effect:<12}: {n:>4} ({pct:.1f}%)')


# ── Error analysis ───────────────────────────────────────────────────────────

print('\n' + '=' * 70)
print('ERROR RATE ANALYSIS')
print('=' * 70)

for model in ['Qwen2VL', 'LLaVA']:
    model_df = df[df['model'] == model]
    b_errors = (model_df['baseline_answer'] == 'ERROR').sum()
    r_errors = (model_df['revised_answer'] == 'ERROR').sum()
    print(f'{model}: baseline errors={b_errors}, revised errors={r_errors}')


# ── Relative error ───────────────────────────────────────────────────────────

print('\n' + '=' * 70)
print('MEAN RELATIVE ERROR')
print('=' * 70)

numerical_types = ['minimum', 'maximum', 'median', 'mean']

def mean_relative_error(predicted_series, gt_series):
    errors = []
    for pred, gt in zip(predicted_series, gt_series):
        p = extract_number(pred)
        g = extract_number(str(gt))
        if p is not None and g is not None and g != 0:
            errors.append(abs(p - g) / abs(g))
    return np.mean(errors) if errors else float('nan')


for model in ['Qwen2VL', 'LLaVA']:
    print(f'\n{model}:')
    model_df = df[df['model'] == model]
    num_df = model_df[model_df['type'].isin(numerical_types)]

    b_mre = mean_relative_error(num_df['baseline_answer'], num_df['ground_truth'])
    r_mre = mean_relative_error(num_df['revised_answer'], num_df['ground_truth'])

    print(f'  Baseline MRE: {b_mre:.3f}')
    print(f'  Revised MRE:  {r_mre:.3f}')
    print(f'  Change:       {r_mre - b_mre:+.3f}')


# ── Save results ─────────────────────────────────────────────────────────────

df.to_csv('results_scored.csv', index=False)
print('Saved results_scored.csv')

summary_rows = []
for model in ['Qwen2VL', 'LLaVA']:
    for qtype in sorted(df['type'].unique()):
        qdf = df[(df['model'] == model) & (df['type'] == qtype)]
        if len(qdf) == 0:
            continue
        summary_rows.append({
            'model': model,
            'question_type': qtype,
            'n': len(qdf),
            'baseline_acc': qdf['baseline_score'].mean(),
            'revised_acc': qdf['revised_score'].mean(),
            'improvement': qdf['revised_score'].mean() - qdf['baseline_score'].mean(),
            'pct_improved': (qdf['dentist_effect'] == 'improved').mean() * 100,
            'pct_worsened': (qdf['dentist_effect'] == 'worsened').mean() * 100,
            'pct_unchanged': (qdf['dentist_effect'] == 'unchanged').mean() * 100,
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('results_summary.csv', index=False)
print('Saved results_summary.csv')
print(summary_df.to_string())