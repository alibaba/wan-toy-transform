<div align="center">
<h2><center>Wan Toy Transform</h2>
<br>
<img src="assets/logo.webp" width="200"/>
<br>
Alibaba Research Intelligence Computing
<br>
<a href="https://github.com/alibaba/wan-toy-transform"><img src='https://img.shields.io/badge/Github-Link-black'></a>
<a href='https://modelscope.cn/models/Alibaba_Research_Intelligence_Computing/wan-toy-transform'><img src='https://img.shields.io/badge/🤖_ModelScope-weights-%23654dfc'></a>
<a href='https://huggingface.co/Alibaba-Research-Intelligence-Computing/wan-toy-transform'><img src='https://img.shields.io/badge/🤗_HuggingFace-weights-%23ff9e0e'></a>
<br>
</div>

This is a LoRA model finetuned on [Wan-I2V-14B-480P](https://github.com/Wan-Video/Wan2.1). It turns things in the image into fluffy toys. 🌟 Give it a star if you like it.

## 🎞️ Showcases

<img src="assets/showcases/1.webp" width="40%"/> <img src="assets/showcases/2.webp" width="40%"/> <img src="assets/showcases/3.webp" width="40%"/> <img src="assets/showcases/4.webp" width="40%"/>

## 🐍 Installation

```bash
# Python 3.12 and PyTorch 2.6.0 are tested.
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 🔄 Inference

```bash
python generate.py --prompt "The video opens with a clear view of a $name. Then it transforms to a b6e9636 JellyCat-style $name. It has a face and a cute, fluffy and playful appearance." --image $image_path --save_file "output.mp4" --offload_type leaf_level
```

Note:

- Change `$name` to the object name you want to transform.
- `$image_path` is the path to the first frame image.
- Choose `--offload_type` from ['leaf_level', 'block_level', 'none', 'model']. More details can be found [here](https://huggingface.co/docs/diffusers/optimization/memory#group-offloading).
- VRAM usage and generation time of different `--offload_type` are listed below.

  | `--offload_type`                     | VRAM Usage | Generation Time (NVIDIA A100) |
  | ------------------------------------ | ---------- | ----------------------------- |
  | leaf_level                           | 11.9 GB    | 17m17s                        |
  | block_level (num_blocks_per_group=1) | 20.5 GB    | 16m48s                        |
  | model                                | 39.4 GB    | 16m24s                        |
  | none                                 | 55.9 GB    | 16m08s                        |

## 🤝 Acknowledgements

Special thanks to these projects for their contributions to the community!

- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)
- [diffusers](https://github.com/huggingface/diffusers)

## 📄 Our previous work

- [Tora: Trajectory-oriented Diffusion Transformer for Video Generation](https://github.com/alibaba/Tora)
- [AnimateAnything: Fine Grained Open Domain Image Animation with Motion Guidance](https://github.com/alibaba/animate-anything)
