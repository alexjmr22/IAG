#!/usr/bin/env python
# coding: utf-8
"""
Gera barcharts FID e KID para todos os experimentos com results.csv.

Uso:
  python scripts/plot_results.py                        # todos os experimentos
  python scripts/plot_results.py --filter diff          # só pastas com 'diff' no nome
  python scripts/plot_results.py --filter prod ema dcgan  # pastas PROD + EMA + DCGAN
  python scripts/plot_results.py --top 10              # só os 10 melhores por FID

Gera em results/:
  - fid_comparison.png
  - kid_comparison.png
  - fid_kid_scatter.png   (FID vs KID por modelo)
"""

from __future__ import annotations
import argparse, csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS   = REPO_ROOT / 'results'

# Cores por família de modelo
FAMILY_COLORS = {
    'Diffusion':    '#4C72B0',
    'DiffusionEMA': '#1a9641',
    'DCGAN':        '#d62728',
    'VAE':          '#ff7f0e',
}

def load_all_results(filter_tags: list[str] | None, top_n: int | None) -> list[dict]:
    rows = []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir(): continue
        csv_path = d / 'results.csv'
        if not csv_path.exists(): continue
        if filter_tags and not any(t.lower() in d.name.lower() for t in filter_tags):
            continue
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows.append({
                    'exp':      d.name,
                    'model':    row['model'],
                    'fid_mean': float(row['fid_mean']),
                    'fid_std':  float(row['fid_std']),
                    'kid_mean': float(row['kid_mean']),
                    'kid_std':  float(row['kid_std']),
                })

    # ordenar por FID crescente
    rows.sort(key=lambda r: r['fid_mean'])
    if top_n:
        rows = rows[:top_n]
    return rows


def bar_chart(rows, metric_mean, metric_std, metric_label, out_path, title):
    labels = [r['exp'] for r in rows]
    means  = [r[metric_mean] for r in rows]
    stds   = [r[metric_std]  for r in rows]
    colors = [FAMILY_COLORS.get(r['model'], '#888888') for r in rows]

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.55), 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.85, ecolor='black', linewidth=0.5)

    # valor em cima de cada barra
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + max(means) * 0.01,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7.5)
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.18)

    # legenda de famílias
    legend_patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()
                      if any(r['model'] == f for r in rows)]
    ax.legend(handles=legend_patches, fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'-> {out_path.name}')


def scatter_fid_kid(rows, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        color = FAMILY_COLORS.get(r['model'], '#888888')
        ax.errorbar(r['fid_mean'], r['kid_mean'],
                    xerr=r['fid_std'], yerr=r['kid_std'],
                    fmt='o', color=color, alpha=0.75, markersize=5, linewidth=0.8)
        ax.annotate(r['exp'], (r['fid_mean'], r['kid_mean']),
                    textcoords='offset points', xytext=(4, 2), fontsize=5.5, alpha=0.8)

    ax.set_xlabel('FID (↓)')
    ax.set_ylabel('KID mean (↓)')
    ax.set_title('FID vs KID — todos os experimentos')
    ax.grid(alpha=0.3)
    legend_patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()
                      if any(r['model'] == f for r in rows)]
    ax.legend(handles=legend_patches, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'-> {out_path.name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', nargs='*', default=None,
                        help='Filtrar pastas que contenham estas strings (ex: --filter prod ema dcgan)')
    parser.add_argument('--top', type=int, default=None,
                        help='Mostrar só os N melhores por FID')
    args = parser.parse_args()

    rows = load_all_results(args.filter, args.top)
    if not rows:
        print('Nenhum results.csv encontrado.'); return

    print(f'{len(rows)} experimentos carregados.')

    suffix = ''
    if args.filter: suffix += '_' + '_'.join(args.filter)
    if args.top:    suffix += f'_top{args.top}'

    bar_chart(rows, 'fid_mean', 'fid_std', 'FID (↓)',
              RESULTS / f'fid_comparison{suffix}.png',
              f'FID por experimento (média ± std, {len(rows)} modelos)')

    bar_chart(rows, 'kid_mean', 'kid_std', 'KID mean (↓)',
              RESULTS / f'kid_comparison{suffix}.png',
              f'KID por experimento (média ± std, {len(rows)} modelos)')

    scatter_fid_kid(rows, RESULTS / f'fid_kid_scatter{suffix}.png')

    # Tabela resumo no terminal
    print(f'\n{"Experimento":<30} {"Modelo":<14} {"FID":>8} {"±":>5} {"KID":>8} {"±":>6}')
    print('-' * 75)
    for r in rows:
        print(f'{r["exp"]:<30} {r["model"]:<14} {r["fid_mean"]:>8.2f} {r["fid_std"]:>5.2f} '
              f'{r["kid_mean"]:>8.4f} {r["kid_std"]:>6.4f}')

if __name__ == '__main__':
    main()
