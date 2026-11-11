import gzip
import csv
import numpy as np
import torch
from pathlib import Path

from gtr.learned_generator import (PairNet, LearnedGenerator, index_pair,
                                   pair_index, encode_context, save_learned, load_learned)
from gtr.learned_train import examples_from_moves, make_dataset, evaluate_nll
from gtr.puyo_generators import PuyoGenerator


def test_pair_codec_and_architecture():
    for i in range(16):
        assert pair_index(index_pair(i))==i
    m=PairNet()
    assert sum(p.numel() for p in m.parameters()) == 5872
    assert torch.allclose(m(torch.tensor([[0,0,0,0],[44,15,15,15]])),torch.zeros(2,16))


def test_autoregression_and_no_hidden_reads():
    g=LearnedGenerator(mode='online',seed=1,batch_size=2,warmup=2,update_every=1,capacity=8)
    class Env:
        def __init__(self):
            self.ob={'turn':4,'visible':[('R','R'),('G','B'),('Y','G')]}
            self.passed=None
        def observation(self): return self.ob
        def fork_with_hidden_pairs(self,p):
            self.passed=p
            return p
    env=Env()
    fork=PuyoGenerator('learned',g,15).fork(env,np.random.default_rng(10))
    assert len(fork)==128
    a=[pair_index(x) for x in env.ob['visible']]
    assert len(fork[7:20])==13
    assert len(fork[20:])==108
    # Never access any hidden queue method: only observation and public fork API.
    assert isinstance(fork[7],str)


def test_online_observe_only_newly_public_label_and_resume(tmp_path):
    before={'turn':1,'visible':[('R','Y'),('G','G'),('B','R')]}
    after={'turn':2,'visible':[('G','G'),('B','R'),('Y','Y')]}
    g=LearnedGenerator(mode='online',seed=5,batch_size=2,warmup=2,update_every=1,capacity=8)
    g.observe(before,after,terminal=True)
    assert g.seen==0
    g.observe(before,after,terminal=False)
    assert g.seen==1 and g.labels[0]==pair_index('YY')
    g.observe(before,after,terminal=False)
    assert g.seen==2 and g.updates==1
    with torch.no_grad():
        probs=g.model(torch.tensor([encode_context(1,before['visible'])]))
    assert torch.all(torch.isfinite(probs))
    state=g.state_dict()
    path=tmp_path/'resume.pt'
    torch.save(state,path)
    loaded=torch.load(path,weights_only=True)
    r=LearnedGenerator(mode='online',seed=5,batch_size=2,warmup=2,update_every=1,capacity=8)
    r.load_state_dict(loaded)
    assert r.seen==2 and r.updates==1
    assert np.array_equal(r.contexts,g.contexts)
    assert r.rng.integers(100)==g.rng.integers(100)
    assert all(torch.equal(r.model.state_dict()[k],g.model.state_dict()[k]) for k in g.model.state_dict())
    save_learned(tmp_path/'weights.pt',g.model)
    frozen=LearnedGenerator(mode='frozen',seed=5,init_checkpoint=tmp_path/'weights.pt')
    assert frozen.optimizer is None
    assert all(torch.equal(frozen.model.state_dict()[k],g.model.state_dict()[k]) for k in g.model.state_dict())


def test_real_move_csv_supervision_no_future_from_same_row(tmp_path):
    p=tmp_path/'train_moves.csv.gz'
    with gzip.open(p,'wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['episode','seed','turn','visible_current','visible_next','visible_next2'])
        w.writeheader()
        for seed in [100,101,102,103]:
            queue=['RR','RY','GB','YB','BG','YY']
            for t in range(3):
                w.writerow({'episode':seed,'seed':seed,'turn':t+1,
                            'visible_current':queue[t],'visible_next':queue[t+1],
                            'visible_next2':queue[t+2]})
    examples=list(examples_from_moves(p))
    assert len(examples)==8
    assert examples[0][1:]==(0,['RR','RY','GB'],'YB')
    train,val,ntrain,nval=make_dataset([p],0,600000,0.25,44)
    assert ntrain==3 and nval==1
    assert len(train[1])==6 and len(val[1])==2
    assert evaluate_nll(PairNet(),val)[0]>2


def test_native_learned_5step_target_isolated():
    try:
        from rl_env import puyo_cpp
    except ImportError:
        import pytest
        pytest.skip('Puyo C++ engine not compiled')
    if not hasattr(puyo_cpp.PuyoEnv,'fork_with_hidden_pairs'):
        import pytest
        pytest.skip('Fork API not present')
    from gtr.reward import make_environment
    from gtr.model_rollout import simulate_target
    from gtr.actions import encode_candidates
    from gtr.replay import td_targets
    class ZeroValue(torch.nn.Module):
        def forward(self,x): return x.new_zeros((len(x),))
    env=make_environment(.995,.04,1000.)
    env.reset(12345)
    original=env.observation()
    candidate=env.preview_all()[0]
    gen=LearnedGenerator(mode='online',seed=101)
    fork=PuyoGenerator('learned',gen,5).fork(env,np.random.default_rng(2))
    assert env.observation()==original
    model=ZeroValue()
    value=simulate_target(
        fork,int(candidate['action']),encode_candidates([candidate])[0],
        chains=int(candidate['chains']),n_step=5,model=model,
        device=torch.device('cpu'),rng=np.random.default_rng(4),epsilon=.4,
        gamma=.995,score_scale=1000.,gameover_penalty=-1000.,infer_chunk=32)
    assert value.first_board==tuple(candidate['observation']['board'])
    assert value.first_score==candidate['score']
    assert len(value.simulated_actions)<=4
    assert np.isfinite(td_targets([value.transition],model,torch.device('cpu'),.995,32)[0])
    assert env.observation()==original
