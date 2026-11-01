"""354: misma observación PPID/cwd de core-reparto, extendida a fases train/BPB/CORE."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import time


def observar(raiz, registros, auxiliar):
    spec=importlib.util.spec_from_file_location('arbol_reparto_base',auxiliar)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    planes=[]
    for p in raiz.glob('*.plan.json'):
        r=json.loads(p.read_text());s=Path(r['sid_file'])
        try:r['sid']=int(s.read_text())
        except (OSError,ValueError):continue
        assert r['sid']>1;planes.append(r)
    procesos={};errores=[];vivos=[];lanzadores=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            c=(p/'stat').read_text().rsplit(')',1)[1].split()
            procesos[int(p.name)]=dict(estado=c[0],ppid=int(c[1]),sid=int(c[3]),inicio_ticks=int(c[19]))
        except (OSError,ValueError,IndexError):continue
    for plan in planes:
        pid=plan['sid'];r=procesos.get(pid);k='launcher:'+str(pid)
        if not r or r['estado'] in ('Z','X') or r['sid']!=pid:continue
        previo=registros.get(k)
        if (raiz/(plan['fase']+'.done')).exists() and not (previo and previo['inicio_ticks']==r['inicio_ticks']):continue
        dato=dict(pid=pid,**r,fase=plan['fase'])
        registros[k]=dato;lanzadores.append(dato)
    for pid,r in procesos.items():
        if r['estado'] in ('Z','X'):continue
        posibles=[]
        for plan in planes:
            previo=registros.get(str(pid));actual=pid;vistos=set()
            while actual in procesos and actual not in vistos:
                vistos.add(actual);n=procesos[actual]
                if n['sid']==plan['sid']:posibles.append(plan);break
                actual=n['ppid']
            else:
                if previo and previo['inicio_ticks']==r['inicio_ticks'] and previo['fase']==plan['fase']:posibles.append(plan)
        if not posibles:continue
        p=Path('/proc',str(pid))
        try:
            argv=(p/'cmdline').read_bytes().split(b'\0');cwd=p.joinpath('cwd').resolve(strict=True)
            env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
            if b'LOCAL_RANK' not in env:continue
            for plan in posibles:
                if cwd!=Path(plan['cwd']).resolve() or plan['argv'].encode() not in argv:continue
                k=str(pid);previo=registros.get(k);actual=pid;vistos=set();asc=False;cadena=[]
                while actual in procesos and actual not in vistos:
                    vistos.add(actual);n=procesos[actual];cadena.append(dict(pid=actual,**n))
                    if n['sid']==plan['sid']:asc=True;break
                    actual=n['ppid']
                conocido=previo and previo['inicio_ticks']==r['inicio_ticks'] and previo['fase']==plan['fase']
                if not asc and not conocido:
                    errores.append(dict(pid=pid,fase=plan['fase'],causa='Worker cwd/argv coincide pero sin ascendencia ni identidad previa'));continue
                c=(p/'stat').read_text().rsplit(')',1)[1].split();assert int(c[19])==r['inicio_ticks']
                dato=dict(pid=pid,**r,fase=plan['fase'],cwd=str(cwd),ancestros=cadena,
                    entorno={k:env.get(k.encode(),b'').decode(errors='replace') for k in (*base.CLAVES,'QF2_MLP_TRAIN_BF16','QF2_MUON_INICIO','QF2_INICIO_DESDE','QF2_MEMHOST_RUTA','QF2_BANCO_LRM')})
                registros[k]=dato;vivos.append(dato)
        except (FileNotFoundError,ProcessLookupError):continue
        except (OSError,ValueError,IndexError,AssertionError) as exc:errores.append(dict(pid=pid,causa=type(exc).__name__))
    return dict(utc_epoch=time.time(),workers=vivos,launchers=lanzadores,registros=registros,errores=errores)


def limpiar_fase(raiz,fase,auxiliar):
    archivo=raiz/'workers-registrados.json';reg=json.loads(archivo.read_text()) if archivo.exists() else {}
    r=observar(raiz,reg,auxiliar);propios=[v for v in r['workers'] if v['fase']==fase]
    for senal in (signal.SIGTERM,signal.SIGKILL):
        for v in propios:
            try:
                p=Path('/proc',str(v['pid']));c=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(c[19])==v['inicio_ticks'] and int(c[3])==v['sid'] and p.joinpath('cwd').resolve()==Path(v['cwd']):os.killpg(v['sid'],senal)
            except (OSError,ValueError,IndexError):continue
        if propios:time.sleep(2)
    despues=observar(raiz,reg,auxiliar)
    (raiz/(fase+'.limpieza.json')).write_text(json.dumps(despues)+'\n')
    assert not [v for v in despues['workers'] if v['fase']==fase]
    assert not [v for v in despues['errores'] if v.get('fase')==fase]
    return len(propios)
