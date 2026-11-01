"""Corte36 de GPT356 por AST original; sin hooks ni edición del participante.

La separación puede CAMBIAR fusión/redondeo Native. CPU PASS no habilita fit:
primero exigir fulloriginal == prefix compilado + suffix compilado como unidad.
"""
import ast
import copy
import hashlib
import inspect
import linecache
import textwrap
import torch
from torch import nn

CUT=36
CACHE_KEYS=('h','x0','cos','sin','idx',*(f've{i}' for i in range(36,40)),*(f'mem{i}' for i in range(36,40)))

def prefix_function(train,cut=CUT):
    source=textwrap.dedent(inspect.getsource(train.GPT.forward))
    f=ast.parse(source).body[0]
    loop=next(n for n in f.body if isinstance(n,ast.For) and ast.unparse(n.iter)=='enumerate(self.transformer.h)')
    pos=f.body.index(loop)
    body=copy.deepcopy(f.body[:pos+1])
    body[-1].iter=ast.parse(f'enumerate(self.transformer.h[:{cut}])',mode='eval').body
    ve=next(n for n in loop.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ve' for t in n.targets))
    class Index(ast.NodeTransformer):
        def __init__(self,i):self.i=i
        def visit_Name(self,n):
            if n.id=='i':return ast.copy_location(ast.Constant(self.i),n)
            if n.id=='ve':return ast.copy_location(ast.Name(f've{self.i}',n.ctx),n)
            return n
    for i in range(cut,40):body.append(Index(i).visit(copy.deepcopy(ve)))
    values=['x','x0','cos','sin','idx',*[f've{i}' for i in range(cut,40)],*['None']*4]
    body.append(ast.Return(ast.parse('('+','.join(values)+')',mode='eval').body))
    f.name='prefix36_original';f.body=body
    tree=ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[]))
    namespace=dict(vars(train));code=ast.unparse(tree)+'\n';filename='<GPT356 prefix36 from frozen AST>'
    linecache.cache[filename]=(len(code),None,code.splitlines(True),filename)
    exec(compile(code,filename,'exec'),namespace)
    return namespace[f.name],dict(original_forward_sha256=hashlib.sha256(source.encode()).hexdigest(),prefix_ast=ast.unparse(tree))

class Prefix(nn.Module):
    def __init__(self,raw,train):
        super().__init__();self.gpt=raw;self._fn,self.source=prefix_function(train)
    def forward(self,idx,mem=None):
        values=self._fn(self.gpt,idx,mem=mem)
        return dict(zip(CACHE_KEYS,values))

class Suffix(nn.Module):
    """Four real blocks + final RMSnorm + original lm_head/FP32/softcap."""
    def __init__(self,raw,train):
        super().__init__()
        self.blocks=nn.ModuleList(list(raw.transformer.h)[CUT:])
        self.lm_head=raw.lm_head;self._norm=train.norm
        self.vocab_size=raw.config.vocab_size;self.softcap=train.LOGIT_SOFTCAP
    def forward(self,h,x0,cos,sin,idx,ve36=None,ve37=None,ve38=None,ve39=None,mem36=None,mem37=None,mem38=None,mem39=None):
        ves=(ve36,ve37,ve38,ve39);mems=(mem36,mem37,mem38,mem39)
        x=h
        for block,ve,mem in zip(self.blocks,ves,mems):x=block(x,x0,cos,sin,ve,mem)
        x=self._norm(x)
        z=self.lm_head(x)[...,:self.vocab_size].float()
        return self.softcap*torch.tanh(z/self.softcap)

class Split:
    def __init__(self,raw,train,cut=CUT):
        c=raw.config
        assert cut==36 and c.n_layer==40
        assert not any((c.registro,c.prefijo_bos,c.memoria,c.ngramas,c.memcorpus,c.pkm,c.mos,c.delta_ctx))
        assert raw.delta is None and raw.pkm is None and raw.mos is None
        if raw.resumen is not None:assert raw.resumen.capa_entrada<cut and raw.resumen.capa_salida<cut
        if c.memhost:assert all(i<cut for i in raw.memhost_cfg.gated)
        assert len(raw.transformer.h)==40
        self.prefix=Prefix(raw,train);self.suffix=Suffix(raw,train)
        self.scope='Eval predictor only; cache preserves VE and x0; no optimizer or eligibility claim'

def save_cache(path,cache):
    """CPU raw tensors, no dtype conversion; owns storage and no live graph."""
    from pathlib import Path
    path=Path(path);assert set(cache)==set(CACHE_KEYS) and not path.exists()
    value={k:v.detach().cpu().contiguous().clone() if isinstance(v,torch.Tensor) else v for k,v in cache.items()}
    assert all(v is None or isinstance(v,torch.Tensor) for v in value.values())
    torch.save(value,path)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
