"""Documentos TRAIN reales, sin packing ni CORE; cache CPU efímera autenticada."""
import hashlib
import json
from pathlib import Path
import time
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps({'shape': list(value.shape), 'dtype': str(value.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')



def verificar_documentos_congelados(manifest, specification):
    """Autentica manifiesto previo y exige exactamente la misma selección."""
    if not isinstance(specification, dict) or set(specification) != {'path', 'sha256'}:
        raise ValueError('frozen_documents_json requiere path y sha256 exactos')
    path = Path(specification['path'])
    digest = sha(path)
    if digest != specification['sha256']:
        raise ValueError('SHA de frozen-documents no coincide')
    frozen = json.loads(path.read_text())
    fields = ('documents', 'source_shards_sha256', 'bos_id', 'fit_count', 'heldout_count', 'shards')
    for field in fields:
        if field not in frozen or manifest.get(field) != frozen[field]:
            raise ValueError('Manifiesto congelado difiere: ' + field)
    if len(frozen['documents']) != 96 or (frozen['fit_count'], frozen['heldout_count']) != (64, 32):
        raise ValueError('Manifiesto congelado debe contener exactamente64+32documentos')
    return {'path': str(path), 'sha256': digest, 'matched': True, 'fields': list(fields)}


def seleccionar(prepare, tokenizer, out, *, check_budget, provenance):
    """Primeros 96 docs elegibles únicos en shards2/3: FIT64, heldout32."""
    rows, seen = [], set()
    scanned = duplicates = 0
    bos = tokenizer.get_bos_token_id()
    for texts in prepare._document_batches_once('train', 0, 1, 128, shard_ids=[2, 3]):
        for text in texts:
            check_budget()
            scanned += 1
            if not isinstance(text, str):
                raise TypeError('Documento TRAIN no textual')
            identity = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if identity in seen:
                duplicates += 1
                continue
            seen.add(identity)
            tokens = tokenizer.encode(text, prepend=bos)
            if len(tokens) < 1025:
                continue
            ids = torch.tensor(tokens[:1025], dtype=torch.int64)
            if bool(((ids < 0) | (ids >= tokenizer.get_vocab_size())).any()):
                raise ValueError('Token fuera de vocabulario')
            rows.append({'document_id': identity, 'document_tokens': len(tokens),
                         'source_ordinal': scanned - 1, 'ids': ids})
            if len(rows) == 96:
                break
        if len(rows) == 96:
            break
    if len(rows) != 96:
        raise RuntimeError('Menos de96documentos elegibles; no cambiar selector')
    sources = {str(p): sha(p) for p in prepare._paths_for_split('train', shard_ids=[2, 3])}
    manifest = {'selection': 'firsteligible>=1025, unique SHA256 UTF8 full original text',
                'shards': [2, 3], 'bos_id': bos, 'fit_count': 64, 'heldout_count': 32,
                'scanned': scanned, 'duplicates': duplicates, 'provenance': provenance,
                'source_shards_sha256': sources, 'documents': []}
    for index, row in enumerate(rows):
        manifest['documents'].append({k: v for k, v in row.items() if k != 'ids'} |
                                     {'split': 'fit' if index < 64 else 'heldout',
                                      'tokens1025_sha256': tensor_sha(row['ids'])})
    if 'frozen_documents_json' in provenance:
        manifest['frozen_documents_verification'] = verificar_documentos_congelados(
            manifest, provenance['frozen_documents_json'])
    write_json(Path(out) / 'documentos.json', manifest)  # Antes de cache y fit.
    return rows[:64], rows[64:], manifest


def construir_cache(documents, prefix, device, out, split, *, sync, check_budget,
                    host_bytes_remaining):
    result, receipts = [], []
    total = 0
    started = time.monotonic()
    for index, doc in enumerate(documents):
        check_budget()
        ids = doc['ids'][:1024].reshape(1, 1024).clone()
        targets = doc['ids'][1:1025].reshape(1, 1024).clone()
        with torch.no_grad():
            generated = prefix(ids.to(device=device, dtype=torch.int32))
            sync()
            cache = {k: v.detach().cpu().contiguous().clone() if isinstance(v, torch.Tensor) else v
                     for k, v in generated.items()}
        if not torch.equal(cache['idx'].long(), ids) or any(v is not None and not isinstance(v, torch.Tensor) for v in cache.values()):
            raise ValueError('Cache no conserva IDs/valores completos')
        size = sum(v.numel() * v.element_size() for v in cache.values() if isinstance(v, torch.Tensor))
        size += ids.numel() * ids.element_size() + targets.numel() * targets.element_size()
        total += size
        if total > host_bytes_remaining:
            raise MemoryError('Cachehost excede límite; no cambiar panel')
        batch = {'cache': cache, 'targets': targets, 'document_ids': [doc['document_id']]}
        result.append(batch)
        receipt = {'document_id': doc['document_id'], 'host_bytes': size,
                   'values': {k: {'sha256': tensor_sha(v), 'shape': list(v.shape), 'dtype': str(v.dtype)}
                              if isinstance(v, torch.Tensor) else None for k, v in cache.items()},
                   'targets_sha256': tensor_sha(targets)}
        if index == 0:
            path = Path(out) / f'cache-{split}-0.pt'
            with path.open('xb') as stream:
                torch.save(batch, stream)
            receipt['sample_file'] = {'path': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size}
        receipts.append(receipt)
    return result, {'split': split, 'host_bytes': total, 'documents': receipts,
                    'wall_seconds': time.monotonic() - started}


def a_device(batch, device):
    return {'cache': {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch['cache'].items()},
            'targets': batch['targets'].to(device), 'document_ids': batch['document_ids']}
