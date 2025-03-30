# 音乐视频合成

根据给定的音乐（含歌词）在给定视频集内剪出合适的片段，进行剪辑、拼接、合成。

## 使用方法

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
```

```sh
pip install tensorflow==2.17.0
apt-get install ffmpeg
pip install ffmpeg-python pillow torchvision
git clone https://github.com/soCzech/TransNetV2.git
cd TransNetV2
git lfs pull
python setup.py install
```

```sh
pip install git+https://github.com/huggingface/transformers accelerate flash-attn
pip install qwen-vl-utils[decord] sentence_transformers openai
```

### 分割视频片段

```sh
python generate_scenes.py
```

### 计算视频片段长度

```sh
python scenes_length.py
```

### 生成视频片段描述

```sh
python generate_descriptions.py
```

### 匹配歌词与视频片段

使用大语言模型询问：

```sh
python match_llm.py
```

或使用Multilingual-E5文本排序：

```sh
python match_sentence_transformers.py
```

或使用GTE文本排序：

```sh
python match_gte_rerank.py
```

### 合成视频

```sh
python compose.py
```

## 鸣谢

本项目用到了以下模型：
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [Multilingual-E5](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
- [TransNet-V2](https://github.com/soCzech/TransNetV2)