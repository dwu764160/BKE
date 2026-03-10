#!/usr/bin/env python3
"""Audit script: aggregate forecast validation and simulation reports.

Usage: python3 scripts/audit_forecast.py
"""
import json
import glob
import os
from math import fabs


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"WARN: failed to load {path}: {e}")
        return None


def gather_season_level(files):
    out = {}
    for fp in files:
        data = load_json(fp)
        if not data:
            continue
        # infer scenario name from filename
        name = os.path.basename(fp)
        if 'preseason_snapshot' in name:
            scenario = 'preseason_snapshot'
        elif 'end_of_season' in name:
            scenario = 'end_of_season'
        else:
            # default file (no suffix) -> assume end_of_season/default
            scenario = 'end_of_season'

        out.setdefault(scenario, {})
        seasons = data.get('seasons', {})
        for season, sdata in seasons.items():
            out[scenario].setdefault(season, {})
            # season-level summary
            ss = sdata.get('season_stats', {})
            if ss:
                out[scenario][season]['season_stats'] = ss
            # per-model
            models = sdata.get('season_stats_by_model', {})
            if models:
                out[scenario][season]['by_model'] = models
            # team-level errors list
            team_results = sdata.get('team_results', [])
            if team_results:
                errs = []
                for t in team_results:
                    team = t.get('team') or t.get('team_abbreviation') or t.get('team')
                    win_error = t.get('win_error')
                    if win_error is None:
                        continue
                    errs.append({'team': team, 'win_error': win_error, 'abs_error': fabs(win_error)})
                errs_sorted = sorted(errs, key=lambda x: x['abs_error'], reverse=True)
                out[scenario][season]['team_errors'] = errs_sorted
    return out


def gather_step2_validation(files):
    out = {}
    for fp in files:
        data = load_json(fp)
        if not data:
            continue
        name = os.path.basename(fp)
        if 'preseason_snapshot' in name:
            scenario = 'preseason_snapshot'
        else:
            scenario = 'end_of_season'
        out.setdefault(scenario, {})
        out[scenario]['overall'] = data.get('overall', {})
        out[scenario]['per_season'] = data.get('per_season', {})
        # compute worst rotation abs errors per scenario
        team_metrics = data.get('team_metrics', [])
        worst = []
        for t in team_metrics:
            pred = t.get('rotation_predicted')
            actual = t.get('rotation_actual_net')
            if pred is None or actual is None:
                continue
            worst.append({'season': t.get('season'), 'team': t.get('team_abbreviation'), 'rot_pred': pred, 'rot_actual': actual, 'abs_err': fabs(pred - actual)})
        worst_sorted = sorted(worst, key=lambda x: x['abs_err'], reverse=True)
        out[scenario]['worst_rotation'] = worst_sorted
    return out


def print_table_season_models(agg):
    print('\n=== Season × Model Performance ===')
    for scenario, seasons in agg.items():
        print(f'\nScenario: {scenario}')
        for season, sdata in seasons.items():
            ss = sdata.get('season_stats')
            if ss:
                print(f'  {season} - overall: MAE={ss.get("mae")}, RMSE={ss.get("rmse")}, r={ss.get("correlation")}')
            bm = sdata.get('by_model', {})
            for model, mvals in bm.items():
                print(f'    {model}: MAE={mvals.get("mae")}, RMSE={mvals.get("rmse")}, r={mvals.get("correlation")}')


def print_top_bottom_teams(agg, top_n=5):
    print('\n=== Team-level Errors (top/bottom by absolute win_error) ===')
    for scenario, seasons in agg.items():
        print(f'\nScenario: {scenario}')
        for season, sdata in seasons.items():
            errs = sdata.get('team_errors', [])
            if not errs:
                continue
            print(f'  {season}:')
            top = errs[:top_n]
            bottom = errs[-top_n:][::-1]
            print('    Worst (largest abs error):')
            for e in top:
                print(f"      {e['team']}: win_error={e['win_error']} (abs={e['abs_error']})")
            print('    Best (smallest abs error):')
            for e in bottom:
                print(f"      {e['team']}: win_error={e['win_error']} (abs={e['abs_error']})")


def print_step2(step2):
    print('\n=== Step-2 Lineup Validation Summary ===')
    for scenario, sdata in step2.items():
        print(f'\nScenario: {scenario}')
        overall = sdata.get('overall', {})
        print(f"  rotation_corr={overall.get('rotation_corr')}, starter_overlap_mean={overall.get('starter_overlap_rate_mean')}, clutch_overlap_mean={overall.get('clutch_overlap_rate_mean')}")
        # per-season rotation
        per = sdata.get('per_season', {})
        for season, vals in per.items():
            print(f"  {season}: rotation_corr={vals.get('rotation_corr')}, starter_overlap={vals.get('starter_overlap_rate_mean')}")
        # worst rotation teams
        worst = sdata.get('worst_rotation', [])[:10]
        if worst:
            print('  Top rotation abs errors:')
            for w in worst[:10]:
                print(f"    {w['season']} {w['team']}: pred={w['rot_pred']:.1f}, actual={w['rot_actual']:.1f}, abs_err={w['abs_err']:.1f}")


def print_projection_validation(val):
    print('\n=== Player Projection Validation (forecast_validation.json) ===')
    if not val:
        print('  (file missing)')
        return
    default = val.get('default_scenario')
    print(f"  default_scenario={default}")
    projections = val.get('projections', {})
    for key, p in projections.items():
        print(f"  {key}: bke_r={p.get('bke_correlation')}, bke_mae={p.get('bke_mae')}, mpg_r={p.get('mpg_correlation')}, mpg_mae={p.get('mpg_mae')}, team_match_rate={p.get('team_match_rate')}")


def main():
    # find files
    season_files = glob.glob('reports/forecast_season_results*.json')
    season_files = sorted(season_files)
    step2_files = glob.glob('reports/forecast_step2_validation*.json')
    val_file = 'reports/forecast_validation.json'

    agg = gather_season_level(season_files)
    step2 = gather_step2_validation(step2_files)
    val = load_json(val_file)

    print_table_season_models(agg)
    print_top_bottom_teams(agg, top_n=5)
    print_step2(step2)
    print_projection_validation(val)


if __name__ == '__main__':
    main()
