import csv
import importlib.util
import json
from pathlib import Path
import pytest

MODULE=Path(__file__).resolve().parents[1]/'analysis'/'horizon2_compare.py'
spec=importlib.util.spec_from_file_location('horizon2_compare',MODULE)
h2=importlib.util.module_from_spec(spec);spec.loader.exec_module(h2)

def generate(path,seeds,bonus):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=h2.REQUIRED);w.writeheader()
        for seed in seeds:
            w.writerow(dict(seed=seed,total_score=1000+bonus,max_chain=4+bonus,
                            turns=50,simulated_candidates=10000,evaluated_candidates=500,
                            decision_time_seconds=25+bonus))

def test_triplet_paired_differences_and_outputs(tmp_path):
    a,b,c=[tmp_path/f'{t}.csv' for t in 'ABC']
    for f,v in [(a,0),(b,1),(c,2)]:generate(f,[100000,100001,100002],v)
    res=h2.compare(a=a,b=b,c=c,out=tmp_path/'out',expected_games=3)
    assert list(res.group)==['A','B','C']
    assert res.iloc[2].mean_max_chain==6
    assert (tmp_path/'out'/'horizon2_paired_differences.csv').exists()
    assert (tmp_path/'out'/'mean_max_chain.png').exists()

def test_seed_mismatch_rejected(tmp_path):
    a,b=tmp_path/'A.csv',tmp_path/'B.csv'
    generate(a,[100000,100001],0);generate(b,[100000,100002],1)
    with pytest.raises(ValueError,match='seed'):
        h2.compare(a=a,b=b,out=tmp_path/'out')

def test_missing_required_column_rejected(tmp_path):
    a=tmp_path/'A.csv';a.write_text('seed,total_score,max_chain\\n100000,100,5\\n')
    with pytest.raises(ValueError):h2.read_episodes(a)
