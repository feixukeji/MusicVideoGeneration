# 音乐视频制作

根据给定歌词在给定视频集内自动选出契合的片段集。

## 使用方法

AutoDL上选择GPU：RTX 3090 24GB，框架：PyTorch

### 环境准备

```sh
mkdir -p /root/autodl-tmp/huggingface
echo 'export HF_HOME=/root/autodl-tmp/huggingface' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### 安装依赖

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

### 准备视频和歌词数据

将视频文件传至`./autodl-tmp/videos/`目录下，支持多种格式（如mp4、mkv等）。

将歌词文件传至`./autodl-tmp/lyrics.lrc`。

### 分割视频片段

```sh
python generate_scenes.py
```

### 生成视频片段描述

```sh
python generate_descriptions.py
```

### 匹配歌词与视频片段

```sh
python match_sentence_transformers.py
```

### 合成视频

根据`./autodl-tmp/best_matches.txt`的内容选择最合适的视频片段手动剪辑合成。

## 鸣谢

本项目用到了以下模型：
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)
- [TransNet-V2](https://github.com/soCzech/TransNetV2)