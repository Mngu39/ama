"""Offline supervised training of the learned stochastic puyo generator.

Data sources:
1) Existing REAL training trajectory CSV (train_moves.csv.gz), which contains
   only public information; label for time t is NEXT2 visible at time t+1.
2) Fresh engine games using public observation() and legal_actions()/step().

There is NO access to any hidden queue or generator source in this module.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F

from .learned_generator import PairNet, encode_context, pair_index, save_learned


def examples_from_moves(path):
    path=Path(path)
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8',newline='') as f:
        rows=csv.DictReader(f)
        prev=None
        for row in rows:
            key=('real_seed',int(row['seed']))
            if prev is not None and key == prev[0] and int(row['turn'])==prev[1]+1:
                old=prev[2]
                visible=[old['visible_current'],old['visible_next'],old['visible_next2']]
                if (visible[1]!=row['visible_current'] or
                        visible[2]!=row['visible_next']):
                    raise ValueError(f'Invalid public pair shift at {path}:{row["turn"]}')
                yield (key,int(old['turn'])-1,visible,row['visible_next2'])
            prev=(key,int(row['turn']),row)


def examples_from_engine(games,seed_start):
    from rl_env import puyo_cpp
    for index in range(games):
        seed=seed_start+index
        env=puyo_cpp.PuyoEnv()
        obs=env.reset(seed)
        for _ in range(50):
            if env.done(): break
            # Choose among legal actions by lowest current column height,
            # to expose new public pairs across many real placements.
            legal=env.legal_actions()
            candidates=[i for i in range(6) if legal[i]]
            if not candidates: candidates=[i for i,x in enumerate(legal) if x]
            if not candidates: break
            board=obs['board']
            heights=[sum(row[col]!='.' for row in board) for col in range(6)]
            action=min(candidates,key=lambda i:(heights[i%6],i))
            result=env.step(action)
            next_obs=result['observation']
            if not (result['terminated'] or result['truncated']):
                before=[''.join(x) for x in obs['visible']]
                after=[''.join(x) for x in next_obs['visible']]
                if before[1:] != after[:2]:
                    raise RuntimeError('Unexpected real visible sequence shift')
                yield (('engine',seed),int(obs['turn']),before,after[2])
            obs=next_obs


def make_dataset(moves,engine_games,engine_seed_start,val_fraction,seed):
    grouped={}
    for filename in moves:
        for key,pos,vis,label in examples_from_moves(filename):
            grouped.setdefault(key,[]).append((encode_context(pos,vis),pair_index(label)))
    if engine_games:
        for key,pos,vis,label in examples_from_engine(engine_games,engine_seed_start):
            grouped.setdefault(key,[]).append((encode_context(pos,vis),pair_index(label)))
    if len(grouped)<2:
        raise ValueError('Need at least TWO independent games for train/validation split')
    keys=list(grouped)
    np.random.default_rng(seed).shuffle(keys)
    n_val=max(1,round(len(keys)*val_fraction))
    train_keys=keys[n_val:]
    val_keys=keys[:n_val]
    if not train_keys:
        raise ValueError('No training games after splitting')
    def flatten(chosen):
        rows=[p for k in chosen for p in grouped[k]]
        if not rows: raise ValueError('Empty training/validation split')
        x=np.array([p[0] for p in rows],dtype=np.int64)
        y=np.array([p[1] for p in rows],dtype=np.int64)
        return torch.from_numpy(x),torch.from_numpy(y)
    return flatten(train_keys),flatten(val_keys),len(train_keys),len(val_keys)


def evaluate_nll(model,data,batch_size=8192):
    x,y=data
    model.eval()
    loss=0.
    correct=0
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            logits=model(x[start:start+batch_size])
            target=y[start:start+batch_size]
            loss+=float(F.cross_entropy(logits,target,reduction='sum').item())
            correct+=int((logits.argmax(1)==target).sum().item())
    return loss/len(y),correct/len(y)


def main(argv=None):
    p=argparse.ArgumentParser(description='Offline supervised learned puyo generator')
    p.add_argument('--moves',type=Path,nargs='*',default=[],help='Real train_moves.csv.gz files')
    p.add_argument('--engine-games',type=int,default=0,
                   help='Optional independent engine games (public observations only)')
    p.add_argument('--engine-seed-start',type=int,default=600000)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--patience',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--validation-fraction',type=float,default=0.1)
    p.add_argument('--seed',type=int,default=77123)
    args=p.parse_args(argv)
    if not args.moves and args.engine_games==0:
        p.error('Supply --moves ... or --engine-games N')
    if args.epochs<1 or args.batch_size<1 or args.patience<1 or args.lr<=0:
        p.error('Invalid optimizer parameters')
    if args.engine_games<0 or not 0<args.validation_fraction<1:
        p.error('Invalid data split')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train,val,ntrain,nval=make_dataset(args.moves,args.engine_games,
                                       args.engine_seed_start,args.validation_fraction,args.seed)
    model=PairNet().cpu()
    optim=torch.optim.Adam(model.parameters(),lr=args.lr)
    prng=torch.Generator().manual_seed(args.seed+2)
    best_loss=float('inf')
    best_epoch=0
    stale=0
    history=[]
    print(f'games train={ntrain} val={nval}; labels train={len(train[1])} val={len(val[1])}',flush=True)
    for epoch in range(1,args.epochs+1):
        model.train()
        indices=torch.randperm(len(train[1]),generator=prng)
        running=0.
        for start in range(0,len(indices),args.batch_size):
            block=indices[start:start+args.batch_size]
            loss=F.cross_entropy(model(train[0][block]),train[1][block])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optim.step()
            running+=float(loss.item())*len(block)
        val_nll,accuracy=evaluate_nll(model,val)
        row={'epoch':epoch,'train_nll':running/len(train[1]),
             'val_nll':val_nll,'val_accuracy':accuracy}
        history.append(row)
        print(f'epoch={epoch} train_nll={row["train_nll"]:.5f} '
              f'val_nll={val_nll:.5f} val_accuracy={accuracy:.3%}',flush=True)
        if val_nll<best_loss-1e-5:
            best_loss=val_nll;best_epoch=epoch;stale=0
            save_learned(args.output,model,metadata={
                'epoch':epoch,'val_nll':val_nll,'val_accuracy':accuracy,
                'train_examples':len(train[1]),'validation_examples':len(val[1]),
                'train_games':ntrain,'validation_games':nval,
                'source_moves':[str(x) for x in args.moves],
                'source_engine_games':args.engine_games,'source_engine_seed_start':args.engine_seed_start,
                'source_engine_seed_end':args.engine_seed_start+args.engine_games-1 if args.engine_games else None,
                'notes':'Model saw only public pairs at t and new NEXT2 at t+1'})
        else:
            stale+=1
        if stale>=args.patience:break
    history_path=args.output.with_suffix('.history.json')
    history_path.parent.mkdir(parents=True,exist_ok=True)
    history_path.write_text(json.dumps({'best_epoch':best_epoch,'best_val_nll':best_loss,
                                        'uniform_nll':float(np.log(16)),
                                        'history':history},indent=2),encoding='utf-8')
    print(f'SAVED {args.output} best_val_nll={best_loss:.5f}; '
          f'uniform_nll={np.log(16):.5f}',flush=True)

if __name__=='__main__':main()
