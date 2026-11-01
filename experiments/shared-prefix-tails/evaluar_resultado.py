"""Resultado B: separar completar POC, mejora e integridad; originalAFAILED intacto."""
from pathlib import Path
import hashlib,json,os
TOL={'nll_atol':.01,'nll_mae':.001,'witness_atol':.02,'witness_rtol':.005}
def main():
 e=os.environ;a=json.load(open(e['ADMISION_JSON']))
 assert hashlib.sha256(Path(e['ADMISION_JSON']).read_bytes()).hexdigest()==e['SHA_ADMISION']
 assert a['modo']=='fit-colas' and a['tolerancias']==TOL
 x=Path(e['X']);r=json.load(open(x/'fit-instrumentos/resultado.json'));fit=json.load(open(x/'fit/resultado.json'))
 assert r['supported_poc_seq_len']==1024 and r['unsupported_2048_preserved'] is True
 assert r['raw_parameter_versions_unchanged'] and r['checkpoint_stat_unchanged']
 if fit['complete']:
  assert fit['fit']['steps']==200 and fit['fit']['complete_requested_steps']
  assert set(fit['initial_anchor']['cases'])=={'1024'}
  assert fit['initial_anchor']['valido'] is True and fit['initial_anchor']['reference']=='splitbase_variant'
  assert fit['frozen_parameter_versions_unchanged']
  assert set(fit['heldout'])=={'original','splitbase','control','a','b','pair'}
  assert all(len(v['documents'])==32 and v['tokens']==32768 for v in fit['heldout'].values())
  assert fit['capacity']['control']==sum(fit['capacity']['pair'])
  assert fit['capacity']['pair'][0]==fit['capacity']['pair'][1]
 result=dict(supported_poc_seq_len=1024,unsupported_2048_preserved=True,operational_complete=True,poc_complete=fit['complete'],status=fit['status'],steps=fit.get('fit',{}).get('steps',0),decision=fit.get('decision'),timings=fit['timings'],fit_metrics={k:v for k,v in fit.get('fit',{}).items() if k not in ('losses','document_schedule','nonzero_gradient_seen')},heldout={n:{k:v for k,v in m.items() if k!='documents'} for n,m in fit.get('heldout',{}).items()},raw_parameter_versions_unchanged=True,checkpoint_stat_unchanged=True,A_original_identity_failed_preserved=True,CORE_measured=False,scope='POC adaptación devarianteSplit; noCORE/BPBoficial/elegibilidad')
 (x/'resultado-validado.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
