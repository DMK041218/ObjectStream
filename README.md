# ObjectStream

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2607.28312-b31b1b.svg)](https://arxiv.org/abs/2607.28312)

Official implementation of
[**ObjectStream: Latent Objects as Memory Anchors for Streaming Video Understanding**](https://arxiv.org/abs/2607.28312).

ObjectStream is a training-free, object-centric visual memory framework for
streaming video understanding. It compresses historical visual tokens into
object-level representations while retaining recent observations and temporal
evidence, without fine-tuning the vision-language model.

## Supported Backbones

Each implementation has its own source code, dependencies, installation guide,
and evaluation entry points. Choose the directory for your backbone:

| Backbone | Implementation and instructions | Environment |
|---|---|---|
| Qwen2.5-VL-7B | [qwen2.5-vl-7b/](qwen2.5-vl-7b/) | Python 3.11 · Transformers 4.50.0 |
| Qwen3.5-9B | [qwen3.5-9b/](qwen3.5-9b/) | Python 3.12 · Transformers 5.18.0 |

Use a separate environment for each implementation. Model checkpoints and
benchmark datasets are obtained separately.

## Repository Structure

```text
ObjectStream/
├── qwen2.5-vl-7b/    # Qwen2.5-VL model, preprocessing, inference, and evaluation
├── qwen3.5-9b/       # Native Qwen3.5 integration, inference, and evaluation
├── LICENSE
└── NOTICE
```

## Getting Started

```bash
git clone https://github.com/DMK041218/ObjectStream.git
cd ObjectStream
```

Then follow the [Qwen2.5-VL-7B guide](qwen2.5-vl-7b/README.md) or
[Qwen3.5-9B guide](qwen3.5-9b/README.md) for installation, inference, and
benchmark evaluation.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for licensing and third-party
attributions.
