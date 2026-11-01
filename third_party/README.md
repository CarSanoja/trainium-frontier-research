# Attribution

The training and inference sources are participant modifications of the
[AWS Neuron Trainium Frontier starter kit](https://github.com/aws-neuron/trainium-frontier).
The upstream Apache-2.0 license and NOTICE were obtained at revision
`2cdad297f0e4cffc39b2f650135a041224e3947e` and are retained at the repository root.
Changes by Team Quanta include model variants, host-memory mechanisms, kernels,
data handling and experiment implementations; these files are not represented
as unmodified upstream baseline code.

The published source imports PyTorch and AWS Neuron/NKI components and follows
model/optimizer designs discussed by their original authors. Those dependencies
are not vendored here and retain their own licenses and installation conditions.
Existing source comments and attributions are preserved byte for byte.

No rights to the training corpus, tokenizer, CORE datasets, private checkpoints,
or external software are granted by this archive's license. Obtain those assets
from their original distributors under the applicable terms.
