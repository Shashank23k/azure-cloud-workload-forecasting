"""Forecast, leakage boundary, malformed input and CLI regression tests."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from cwfc.backtest import evaluate, origins
from cwfc.data import load_training_bundle
from cwfc.models import CANDIDATES, forecast
from cwfc.predict import generate_predictions, run, validate_predictions
from cwfc.validation import rolling_origin_splits


def make_data(path, n=40, horizon=4, future=95.):
    path.mkdir(exist_ok=True)
    readings = pd.DataFrame([
        dict(timestamp=i*300, vm_id=vm, min_cpu=value, avg_cpu=value, max_cpu=value)
        for i in range(n)
        for vm, value in [('a', 10+np.sin(i/3) if i < n-horizon else future),
                          ('b', 20+np.cos(i/4) if i < n-horizon else future)]
    ])
    readings.to_csv(path/'vm_cpu_readings.csv.gz', index=False)
    pd.DataFrame(dict(vm_id=['a','b'], vm_category=['Interactive','Unknown'],
                      virtual_core_bucket=[2,4], memory_gb_bucket=[4,8],
                      avg_cpu=[future,future], timestamp_vm_deleted=[999999]*2)).to_csv(
                          path/'vm_metadata.csv.gz', index=False)
    return readings


@pytest.mark.parametrize('model', CANDIDATES)
@pytest.mark.parametrize('target,value', [('active_vm_count',7.),('aggregate_cpu_utilization',12.)])
def test_constant_and_horizon(model, target, value):
    pred = forecast(np.repeat(value,30),24,model,target)
    assert pred.shape == (24,)
    np.testing.assert_allclose(pred,value,atol=1e-10)


@pytest.mark.parametrize('model', CANDIDATES)
def test_bounds_and_recursive_repeatability(model):
    values = np.linspace(2,99,50)
    pred = forecast(values,24,model,'aggregate_cpu_utilization')
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 100)).all()
    np.testing.assert_array_equal(pred,forecast(values,24,model,'aggregate_cpu_utilization'))


def test_official_folds_and_zero_baseline():
    assert origins(110,24) == [50,62,74,86]
    train = pd.DataFrame(dict(timestamp=np.arange(110)*300, active_vm_count=7., aggregate_cpu_utilization=12.))
    detail, summary, chosen = evaluate(train)
    assert chosen == dict(active_vm_count='persistence',aggregate_cpu_utilization='persistence')
    assert (summary.mae < 1e-12).all() and summary.relative_mae.isna().all()
    assert detail.groupby(['target','model','origin']).size().eq(24).all()


def test_future_poisoning_and_regeneration(tmp_path):
    make_data(tmp_path)
    first, models = run(tmp_path,tmp_path/'first.csv',4)
    make_data(tmp_path,future=-999.)  # Even invalid future labels must not enter fit or diagnostics.
    second, other = run(tmp_path,tmp_path/'second.csv',4)
    pd.testing.assert_frame_equal(first,second)
    assert models == other
    assert (tmp_path/'first.csv').read_bytes() == (tmp_path/'second.csv').read_bytes()
    # Future text must not change the training target dtype or break selection.
    cpu = pd.read_csv(tmp_path/'vm_cpu_readings.csv.gz')
    for column in ['min_cpu', 'avg_cpu', 'max_cpu']:
        cpu[column] = cpu[column].astype(object)
        cpu.loc[cpu.timestamp >= 36*300, column] = 'redacted'
    cpu.to_csv(tmp_path/'vm_cpu_readings.csv.gz', index=False)
    redacted, redacted_models = run(tmp_path, tmp_path/'redacted.csv', 4)
    pd.testing.assert_frame_equal(first, redacted)
    assert models == redacted_models
    result = subprocess.run([sys.executable,'-m','cwfc.predict','--data',str(tmp_path),
                             '--output',str(tmp_path/'cli.csv'),'--horizon','4'],
                            capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path/'cli.csv').read_bytes() == (tmp_path/'first.csv').read_bytes()


def test_missing_cpu_and_shuffled_order(tmp_path):
    cpu = make_data(tmp_path)
    cpu.loc[(cpu.timestamp == 300) & (cpu.vm_id == 'a'),'avg_cpu'] = np.nan
    cpu.sample(frac=1,random_state=7).to_csv(tmp_path/'vm_cpu_readings.csv.gz',index=False)
    bundle = load_training_bundle(tmp_path,train_steps=None,horizon=4)
    row = bundle.targets.loc[bundle.targets.timestamp == 300].iloc[0]
    assert row.active_vm_count == 1
    assert row.aggregate_cpu_utilization == pytest.approx(20+np.cos(1/4))
    assert bundle.targets.timestamp.is_monotonic_increasing
    assert bundle.audit['training_excluded_null_avg_cpu'] == 1
    assert set(bundle.metadata.columns) == {'vm_id','vm_category','virtual_core_bucket','memory_gb_bucket'}


def test_missing_vm_row_is_not_imputed(tmp_path):
    cpu = make_data(tmp_path)
    cpu = cpu.loc[~((cpu.timestamp == 600) & (cpu.vm_id == 'b'))]
    cpu.to_csv(tmp_path/'vm_cpu_readings.csv.gz',index=False)
    bundle = load_training_bundle(tmp_path,train_steps=None,horizon=4)
    assert bundle.targets.loc[bundle.targets.timestamp == 600,'active_vm_count'].item() == 1


@pytest.mark.parametrize('bad', ['null_all','range','infinite_time','bad_cadence','metadata','columns'])
def test_malformed_inputs(tmp_path,bad):
    cpu = make_data(tmp_path)
    if bad == 'null_all': cpu.loc[cpu.timestamp == 300,'avg_cpu'] = np.nan
    if bad == 'range': cpu.loc[0,'avg_cpu'] = 101
    if bad == 'infinite_time':
        cpu.timestamp = cpu.timestamp.astype(float)
        cpu.loc[0,'timestamp'] = np.inf
    if bad == 'bad_cadence': cpu.timestamp *= 2
    if bad == 'columns': cpu = cpu.drop(columns='avg_cpu')
    if bad == 'metadata':
        pd.DataFrame(dict(vm_id=['a','a'],vm_category=['x']*2,
                          virtual_core_bucket=[2]*2,memory_gb_bucket=[4]*2)).to_csv(
                              tmp_path/'vm_metadata.csv.gz',index=False)
    cpu.to_csv(tmp_path/'vm_cpu_readings.csv.gz',index=False)
    with pytest.raises(ValueError): load_training_bundle(tmp_path,train_steps=None,horizon=4)


def test_minimum_and_schema(tmp_path):
    make_data(tmp_path,n=12,horizon=4)
    with pytest.raises(ValueError,match='training timestamps'): run(tmp_path,tmp_path/'p.csv',4)
    valid = pd.DataFrame(dict(timestamp=[300,600],active_vm_count=[1.,2.],aggregate_cpu_utilization=[10.,20.]))
    validate_predictions(valid,[300,600])
    for invalid in [valid.assign(active_vm_count=-1),valid.assign(aggregate_cpu_utilization=101),
                    valid.assign(aggregate_cpu_utilization=np.nan),valid.iloc[::-1],valid.drop(columns='timestamp')]:
        with pytest.raises(ValueError): validate_predictions(invalid,[300,600])


def test_splits_reject_bad_step_and_preserve_seconds():
    train = pd.DataFrame(dict(timestamp=np.arange(20)*300))
    assert rolling_origin_splits(train,horizon=4,min_train_size=8)[0].train_end == 2100
    with pytest.raises(ValueError): rolling_origin_splits(train,horizon=4,min_train_size=8,step=0)


def test_simplicity_tiebreak_cannot_bypass_improvement_gate(monkeypatch):
    """A simpler near-best model must itself clear the documented 5% hurdle."""
    def candidate(values, horizon, model, target):
        error = {'persistence': 1., 'mean': .958, 'moving_average_6': .949}.get(model, 2.)
        return np.repeat(float(values.iloc[-1]) + error, horizon)

    monkeypatch.setattr('cwfc.backtest.forecast', candidate)
    train = pd.DataFrame(dict(timestamp=np.arange(110)*300,
                              active_vm_count=7., aggregate_cpu_utilization=12.))
    _, summary, chosen = evaluate(train)
    assert set(chosen.values()) == {'moving_average_6'}
    selected = summary[summary.model == 'moving_average_6']
    assert (selected.improvement_pct > 5).all()
