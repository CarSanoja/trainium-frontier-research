"""132352: Split es variante nueva; AoriginalFAILED se conserva, nunca rescata."""
from pathlib import Path
import hashlib,json,os,sys,time
START=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def stat(p):
 s=Path(p).stat();return dict(bytes=s.st_size,ino=s.st_ino,mtime_ns=s.st_mtime_ns)
def main():
 E=os.environ;assert sha(E['ADMISION_JSON'])==E['SHA_ADMISION'];a=json.load(open(E['ADMISION_JSON']))
 assert a['financiado'] is True and a['modo']=='fit-colas'
 assert E['NEURON_COMPETITION_R2_CACHE_DIR']==a['data_cache_dir']
 out=Path(E['X'])/'fit-instrumentos';out.mkdir(exist_ok=False)
 r=dict(supported_poc_seq_len=1024,unsupported_2048_preserved=True,complete=False,graphs=[],cases={},timings={},fixture=False,full_GPT=True,timing_scope='Primary model plus scorer-equivalent finite synchronization; fullCPU transfer separately. No CE/CORE latency claim')
 def save():
  r['seconds']=time.monotonic()-START;p=out/'resultado.json';q=out/'resultado.tmp';q.write_text(json.dumps(r,indent=2)+'\n');q.replace(p)
 ck=Path(a['checkpoint']);assert stat(ck)==a['checkpoint_stat'] # SHA completo autenticado una vez por preflight raíz.
 participant=Path(a['participant']);kit=Path(a['kit'])
 assert sha(participant/'train.py')=='48841f03178fd3ac6ca99524762c32094f527e3785322a4b54e6fd098c16a158'
 assert sha(participant/'inference.py')=='40cace5e521e271643c7352de067a53ecef046bc3a05e007ea029cbd601d3275'
 for p,h in a['sources_sha256'].items():assert sha(p)==h
 vpath=Path(a['validacion_json']);assert sha(vpath)==a['validacion_sha256']
 v=json.loads(vpath.read_text());old=Path(a['old_x']);meta=old/'original-base/resultado.json'
 inv=v['inventario_arrays_y_recibos'];assert sha(meta)==inv['original-base/resultado.json']['sha256']
 refs=json.loads(meta.read_text())['case_files'];anchors={}
 for length in (1024,):
  info=refs[f'b1-t{length}'];p=old/'original-base'/info['path'];assert sha(p)==info['sha256']==inv[str(p.relative_to(old))]['sha256'];anchors[length]=p
 for key in ('gate_a','terminal_a','cierre_a'):
  assert sha(a[key+'_json'])==a[key+'_sha256']
 gate_a=json.load(open(a['gate_a_json']));terminal=json.load(open(a['terminal_a_json']));closed=json.load(open(a['cierre_a_json']))
 assert a['prerregistro']=='20261101T144559Z'
 assert gate_a['numeric_pass'] is False and gate_a['fit_equivalence_gate_pass'] is False and gate_a['fit_executed'] is False
 assert gate_a['parameter_versions_unchanged'] and gate_a['checkpoint_stat_unchanged']
 case_a=gate_a['cases']['1024'];assert case_a['cache_reload_exact'] and case_a['branchcopy_exact']
 assert all(case_a['branch_independence'][k] is True for k in ('mutated_a_changed','copy_b_unchanged','raw_suffix_unchanged','finite','a_restored_exact'))
 assert terminal['Status']=='Success' and terminal['ResponseCode']==0
 assert closed['rc']==0 and not any(closed[k] for k in ('fd_residuales','modelos_residuales','errores_global'))
 assert not any(closed['workers_final'][k] for k in ('workers','launchers','errores'))
 assert sha(a['backend_gate_json'])==a['backend_gate_sha256']
 backend_gate=json.load(open(a['backend_gate_json']))
 assert backend_gate['valido'] is True and backend_gate['backend_complete'] is True
 assert all(backend_gate['B0_exact'][k] is True for k in ('baseline','control','a','b'))
 assert backend_gate['frozen_unchanged'] is True and backend_gate['checkpoint_stat_unchanged'] is True and len(backend_gate['updates'])==3
 frozen_documents=a['frozen_documents_json'];assert set(frozen_documents)=={'path','sha256'}
 assert sha(frozen_documents['path'])==frozen_documents['sha256']
 assert a['numeric_tolerances']==dict(logits_atol=.01,logits_rtol=.005)
 sys.path[:0]=[str(kit),str(participant)]
 import torch
 from core_eval import score
 torch.set_num_threads(1);torch.set_grad_enabled(True);torch.manual_seed(1);device=score.init_runtime('neuron')
 anchor_data={t:torch.load(p,weights_only=True,map_location='cpu') for t,p in anchors.items()}
 inputs=anchor_data[1024]['input_ids']
 assert inputs.ndim==2 and inputs.shape[1]>=1024
 inputs=inputs[:1].clone().contiguous()
 module=score.import_inference_module(participant/'inference.py')
 begin=time.monotonic();model=module.load_for_eval(str(ck),device);r['timings']['load_seconds']=time.monotonic()-begin
 raw=model.gpt[0];raw.eval().requires_grad_(False)
 assert raw.config.n_layer==40 and raw.config.n_embd==1536
 from split import Split
 import copy
 from fit_driver import ejecutar_poc
 split=Split(raw,module.train);split.prefix.eval();split.suffix.eval()
 (out/'prefix_generado.py').write_text(split.prefix.source['prefix_ast']+'\n')
 torch._dynamo.config.cache_size_limit=max(torch._dynamo.config.cache_size_limit,64)
 r['dynamo_cache_size_limit']=torch._dynamo.config.cache_size_limit
 from torch._dynamo.backends.registry import lookup_backend
 backend=lookup_backend('neuron');graphs=[]
 def compiler(label):
  def call(gm,example_inputs):
   code=gm.code;tag=f'graph{len(graphs):04d}-{label}'
   (out/(tag+'.py')).write_text(code)
   info=dict(label=label,code_sha256=hashlib.sha256(code.encode()).hexdigest(),status='entered')
   graphs.append(info);(out/'graphs.json').write_text(json.dumps(graphs,indent=2)+'\n')
   try:result=backend(gm,example_inputs);info['status']='returned';return result
   except BaseException as exc:info.update(status='failed',error=repr(exc));raise
   finally:(out/'graphs.json').write_text(json.dumps(graphs,indent=2)+'\n')
  return call
 compiled_prefix=torch.compile(split.prefix,backend=compiler('prefix36'),dynamic=False,fullgraph=True)
 whole=torch.compile(raw,backend=compiler('whole_anchor'),dynamic=False,fullgraph=False)
 suffix_backend=compiler('suffix_fit')
 def prefix_callable(idx):
  assert tuple(idx.shape)==(1,1024),'POC supported length only1024'
  with torch.no_grad():
   cache=compiled_prefix(idx,raw.memoria_externa(idx))
   return {k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in cache.items()}
 def suffix_factory():return copy.deepcopy(split.suffix).eval().requires_grad_(False)
 def compile_suffix(tail):return torch.compile(tail,backend=suffix_backend,dynamic=False,fullgraph=True)
 def sync():module.train.synchronize(device)
 def cpu_output(tail,cache):
  assert tuple(cache['idx'].shape)==(1,1024),'POC output length only1024'
  with torch.enable_grad():z=tail(**cache)
  assert bool(torch.isfinite(z).all().cpu().item())
  return z.detach().cpu()
 split_reference=torch.compile(split.suffix,backend=suffix_backend,dynamic=False,fullgraph=True)
 def original_logits(idx):
  assert tuple(idx.shape)==(1,1024),'POC original reference length only1024'
  with torch.no_grad():return whole(idx,mem=raw.memoria_externa(idx))
 def validar_inicial(models,_first_batch):
  receipt=dict(valido=False,reference='splitbase_variant',A_original_equivalence_passed=False,native_storage_verified=False,cases={},independence={})
  for length in (1024,):
   idx=inputs[:,:length].to(device=device,dtype=torch.int32)
   with torch.no_grad():
    original=original_logits(idx).detach().cpu()
    cache=prefix_callable(idx)
   assert torch.isfinite(original).all()
   outputs={name:cpu_output(tail,cache) for name,tail in models.items()}
   ref=cpu_output(split_reference,cache);measures={};passed=True
   for name,z in outputs.items():
    anchor=anchor_data[length];tail=z[:,length-17:length-1].reshape(16,32768)
    err=(torch.nn.functional.cross_entropy(tail,anchor['targets'],reduction='none')-anchor['nll']).abs()
    good=torch.equal(z,ref)
    measures[name]=dict(passed=good,split_byte_exact=good,original_diagnostic_only=dict(nll_max=float(err.max()),nll_mae=float(err.mean()),full_logits_max=float((z-original).abs().max())))
    passed=passed and good
   isolations={}
   for arm in ('control','a','b'):
    target=next(p for name,p in models[arm].named_parameters() if p.requires_grad and name.endswith('.b'))
    before=target.detach().clone()
    try:
     with torch.no_grad():target.add_(.125)
     changed=cpu_output(models[arm],cache)
     others={name:torch.equal(cpu_output(tail,cache),outputs[name]) for name,tail in models.items() if name!=arm}
     split_unchanged=torch.equal(cpu_output(split_reference,cache),ref)
     isolation=dict(own_changed=not torch.equal(changed,outputs[arm]),others_unchanged=all(others.values()),others=others,split_unchanged=split_unchanged)
    finally:
     with torch.no_grad():target.copy_(before)
    isolation['restored']=torch.equal(cpu_output(models[arm],cache),outputs[arm])
    passed=passed and isolation['own_changed'] and isolation['others_unchanged'] and isolation['split_unchanged'] and isolation['restored']
    isolations[arm]=isolation
   receipt['cases'][str(length)]=measures;receipt['independence'][str(length)]=isolations
   (out/'initial-anchor.json').write_text(json.dumps(receipt,indent=2)+'\n')
   if not passed:return receipt
  receipt['valido']=True;(out/'initial-anchor.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
 before_versions={name:p._version for name,p in raw.named_parameters()}
 started=float(E['OP_RAIZ_MONO']);tokenizer=module.prepare.ensure_tokenizer(build_if_missing=False)
 try:
  variant_authorization=dict(mode='split_variant_vs_original',A_original_equivalence_passed=False,A_cache_copy_passed=True,new_preregister='20261101T144559Z')
  result=ejecutar_poc(prefix_callable,suffix_factory,compile_suffix,sync,device,tokenizer,module.prepare,Path(E['X'])/'fit',started,validar_inicial=validar_inicial,provenance=dict(admision_sha256=E['SHA_ADMISION'],gate_a_sha256=a['gate_a_sha256'],checkpoint_sha256=a['checkpoint_sha256'],backend_gate_sha256=a['backend_gate_sha256'],frozen_documents_json=frozen_documents),original_logits=original_logits,variant_authorization=variant_authorization,rank=4,lr=.001,max_seconds=1020)
  r.update(complete=result['complete'],status=result['status'],fit_result='fit/resultado.json')
 except BaseException as exc:r['error']=repr(exc);raise
 finally:
  r['raw_parameter_versions_unchanged']=all(p._version==before_versions[name] for name,p in raw.named_parameters())
  r['checkpoint_stat_unchanged']=stat(ck)==a['checkpoint_stat'];r['graphs']=graphs;save()
  (Path(E['X'])/'resultado-validado.json').write_text(json.dumps(dict(valido=r.get('complete',False) and r['raw_parameter_versions_unchanged'] and r['checkpoint_stat_unchanged'],status=r.get('status','error'),scope='POC TRAIN+document-heldout; noCORE/finaleligible ni otra financiación'),indent=2)+'\n')
if __name__=='__main__':main()
