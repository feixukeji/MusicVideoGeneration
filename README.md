# Music Video Generation

Automatically select matching video clips from a given video collection based on lyrics.

## Usage

On AutoDL, select GPU: RTX 3090 24GB, Framework: PyTorch

### Environment Setup

```sh
mkdir -p /root/autodl-tmp/huggingface
echo 'export HF_HOME=/root/autodl-tmp/huggingface' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### Install Dependencies

```sh
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
python -m pip install --upgrade pip
```

```sh
apt-get install ffmpeg
pip install ffmpeg-python pillow torchvision
pip install transnetv2-pytorch
```

```sh
pip install git+https://github.com/huggingface/transformers accelerate flash-attn
pip install qwen-vl-utils[decord] sentence_transformers openai
```

### Prepare Video and Lyrics Data

Upload video files to `./autodl-tmp/videos/` directory, supporting multiple formats (e.g., mp4, mkv).

Upload lyrics file to `./autodl-tmp/lyrics.lrc`.

### Split Video into Scenes

```sh
python generate_scenes.py
```

### Generate Scene Descriptions

```sh
python generate_descriptions.py
```

### Match Lyrics with Video Clips

```sh
python match_sentence_transformers.py
```

### Compose Video

Manually edit and compose the video based on the content in `./autodl-tmp/best_matches.txt`.

## Acknowledgements

This project uses the following models:
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)
- [TransNet-V2](https://github.com/soCzech/TransNetV2)