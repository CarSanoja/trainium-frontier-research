"""Predictor diagnóstico posthoc: no constituye un modelo entrenado con la caché."""
import os
import torch
from cache_prefijo import aplicar


class Predictor(torch.nn.Module):
    def __init__(self, base, beta, bos):
        super().__init__()
        self.base, self.beta, self.bos = base, beta, bos
        self.max_seq_len = base.max_seq_len

    def forward(self, input_ids):
        logits = self.base(input_ids)
        return aplicar(input_ids, logits, self.beta, self.bos)

    def get_eval_worker(self):
        return self.base.get_eval_worker()


def load_for_eval(checkpoint_path, device):
    import inference
    import prepare
    beta = float(os.environ.get('CACHE_PREFIJO_BETA', '0'))
    if beta not in (0., 1.):
        raise ValueError('El diagnóstico registrado sólo admite beta0/control o beta1')
    base = inference.load_for_eval(checkpoint_path, device)
    bos = prepare.ensure_tokenizer(build_if_missing=False).get_bos_token_id()
    return Predictor(base, beta, bos).eval()
