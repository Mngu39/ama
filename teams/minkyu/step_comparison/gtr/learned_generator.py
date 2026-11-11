"""Small causal learned Ama pair generator (no access to actual hidden queue).

At turn t, predicts pair z[t+3] from (t,z[t],z[t+1],z[t+2]).
Online labels become available only after the real environment advances to t+1.
The model produces a categorical distribution; future pairs are sampled, never argmaxed.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

COLORS = 'RYGB'
NUM_PAIRS = 16
MAX_POSITION = 64
FORMAT = 1


def pair_index(pair):
    if isinstance(pair, (tuple, list)):
        pair = ''.join(pair)
    if not isinstance(pair, str) or len(pair) != 2 or any(c not in COLORS for c in pair):
        raise ValueError(f'Invalid two-color pair: {pair!r}')
    return COLORS.index(pair[0])*4 + COLORS.index(pair[1])


def index_pair(index):
    index = int(index)
    if not 0 <= index < NUM_PAIRS:
        raise ValueError('Pair index must be between 0 and 15')
    return COLORS[index//4] + COLORS[index%4]


def encode_context(turn, visible):
    if not 0 <= int(turn) < MAX_POSITION:
        raise ValueError(f'Unsupported turn {turn}; max={MAX_POSITION-1}')
    if len(visible) != 3:
        raise ValueError('Expected exactly CURRENT/NEXT/NEXT2')
    return [int(turn), *(pair_index(pair) for pair in visible)]


class PairNet(nn.Module):
    """Position-wise learned baseline logits + short-history residual MLP.

    64*16 + 16*8 + 32*64+64 + 64*32+32 + 32*16+16 = 5872 parameters.
    Baseline and residual are learned solely from observed training sequences.
    """
    def __init__(self):
        super().__init__()
        self.position_logits = nn.Embedding(MAX_POSITION, NUM_PAIRS)
        self.pair_embedding = nn.Embedding(NUM_PAIRS, 8)
        self.residual = nn.Sequential(nn.Linear(32,64), nn.ReLU(),
                                      nn.Linear(64,32), nn.ReLU(),
                                      nn.Linear(32, NUM_PAIRS))
        nn.init.zeros_(self.position_logits.weight)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, contexts):
        contexts = contexts.long()
        pos = contexts[:,0]
        embedded = self.pair_embedding(contexts[:,1:]).reshape(-1,24)
        # Eight fixed, normalized position features; no hidden state / exact RNG seed.
        angle = pos.float().unsqueeze(1)
        frequencies = torch.tensor([1., 2., 4., 8.], device=pos.device).view(1,4)
        scaled = angle / 50. * frequencies * torch.pi
        pos_embed = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)
        x = torch.cat([pos_embed,embedded], dim=1)
        return self.position_logits(pos) + self.residual(x)


def model_probs(model, context):
    model.eval()
    with torch.inference_mode():
        arr = torch.tensor([context], dtype=torch.long)
        p = torch.softmax(model(arr), dim=-1)[0].numpy().astype(np.float64)
    p = p / p.sum()
    if not np.all(np.isfinite(p)):
        raise FloatingPointError('Nonfinite learned generator probabilities')
    return p


def save_learned(path, model, metadata=None):
    # Same atomic saver as the RL run; keep data compatible with weights_only=True.
    from .checkpoints import _atomic_torch_save
    _atomic_torch_save({'format':FORMAT,'model':model.state_dict(),
                        'meta':dict(metadata or {})}, path)


def load_learned(path, model=None):
    data = torch.load(Path(path), map_location='cpu', weights_only=True)
    if data.get('format') != FORMAT or not isinstance(data.get('model'),dict):
        raise ValueError('Unsupported learned generator checkpoint')
    model = PairNet() if model is None else model
    model.load_state_dict(data['model'], strict=True)
    model.eval()
    return model, data.get('meta',{})


class LearnedGenerator:
    """CPU generator; optimizer optional. Model and its RNG are separate from RL."""
    def __init__(self, *, mode, seed, init_checkpoint=None, lr=1e-3,
                 batch_size=128, warmup=128, update_every=8, capacity=30000):
        if mode not in ('online','frozen'):
            raise ValueError('learned mode must be online or frozen')
        if mode == 'frozen' and init_checkpoint is None:
            raise ValueError('frozen learned generator requires --learned-checkpoint')
        self.mode = mode
        self.model = PairNet().cpu()
        self.model.eval()
        self.checkpoint = str(init_checkpoint) if init_checkpoint else None
        if init_checkpoint:
            load_learned(init_checkpoint, self.model)
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9101]))
        self.batch_size=int(batch_size)
        self.warmup=int(warmup)
        self.update_every=int(update_every)
        self.capacity=int(capacity)
        self.seen=0
        self.updates=0
        self.last_loss=None
        self.contexts=np.zeros((capacity,4),dtype=np.int64)
        self.labels=np.zeros(capacity,dtype=np.int64)
        self.optimizer = (torch.optim.Adam(self.model.parameters(),lr=lr)
                          if mode=='online' else None)

    def sample_hidden(self, turn, visible, count, rng):
        """Draw only needed n-step pairs. Subsequent predictions use earlier draws."""
        context=encode_context(turn,visible)
        out=[]
        for _ in range(int(count)):
            if context[0] >= MAX_POSITION:
                raise ValueError('Exceeded learned generator supported positions')
            probs=model_probs(self.model,context)
            sampled=int(rng.choice(NUM_PAIRS,p=probs))
            out.append(index_pair(sampled))
            context=[context[0]+1,context[2],context[3],sampled]
        return out

    def observe(self, before, after, *, terminal):
        """Train only on a newly disclosed real NEXT2, after virtual target is fixed."""
        if self.mode != 'online' or terminal:
            return None
        if int(after['turn']) != int(before['turn'])+1:
            raise RuntimeError('Learned label: unexpected real turn increment')
        a=[pair_index(x) for x in before['visible']]
        b=[pair_index(x) for x in after['visible']]
        if a[1:] != b[:2]:
            raise RuntimeError('Learned label: public pair sequence shifted inconsistently')
        ctx=encode_context(before['turn'], before['visible'])
        idx=self.seen%self.capacity
        self.contexts[idx]=ctx
        self.labels[idx]=b[2]
        self.seen+=1
        if (self.seen >= max(self.warmup,self.batch_size)
                and self.seen%self.update_every==0):
            count=min(self.seen,self.capacity)
            indexes=self.rng.integers(count,size=self.batch_size)
            xt=torch.from_numpy(self.contexts[indexes].copy())
            yt=torch.from_numpy(self.labels[indexes].copy())
            self.model.train()
            loss=F.cross_entropy(self.model(xt),yt)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=5.)
            self.optimizer.step()
            self.model.eval()
            self.updates+=1
            self.last_loss=float(loss.detach().item())
        return self.last_loss

    def state_dict(self):
        return {
            'format':FORMAT,'mode':self.mode,'model':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict() if self.optimizer else None,
            'rng':self.rng.bit_generator.state,'seen':self.seen,
            'updates':self.updates,'last_loss':self.last_loss,
            'contexts':torch.from_numpy(self.contexts.copy()),
            'labels':torch.from_numpy(self.labels.copy()),
            'capacity':self.capacity,'batch_size':self.batch_size,
            'warmup':self.warmup,'update_every':self.update_every,
        }

    def load_state_dict(self,state):
        if state.get('format')!=FORMAT or state.get('mode')!=self.mode:
            raise ValueError('Learned resume format/mode mismatch')
        for key in ('capacity','batch_size','warmup','update_every'):
            if state[key]!=getattr(self,key):
                raise ValueError(f'Learned resume config mismatch: {key}')
        self.model.load_state_dict(state['model'],strict=True)
        if self.optimizer:
            self.optimizer.load_state_dict(state['optimizer'])
        self.rng.bit_generator.state=state['rng']
        self.seen=int(state['seen'])
        self.updates=int(state['updates'])
        self.last_loss=state['last_loss']
        self.contexts=state['contexts'].numpy().copy()
        self.labels=state['labels'].numpy().copy()
        self.model.eval()
