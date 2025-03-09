# 音乐视频合成

根据给定的音乐（含歌词）在给定视频集内剪出合适的片段，进行剪辑、拼接、合成。

```sh
mkdir -p /root/autodl-tmp/huggingface
echo 'export HF_HOME=/root/autodl-tmp/huggingface' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

```sh
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
```

```sh
pip install tensorflow==2.17.0
apt-get install ffmpeg
pip install ffmpeg-python pillow
git clone https://github.com/soCzech/TransNetV2.git
cd TransNetV2
git lfs pull
python setup.py install
```

```sh
pip install git+https://github.com/huggingface/transformers accelerate flash-attn
pip install qwen-vl-utils[decord]
```

```sh
pip install openai
```

```sh
python generate_scenes.py
python scenes_length.py
python generate_descriptions.py
python match_llm.py
python compose.py
```

本项目用到了以下模型：
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [TransNetV2](https://github.com/soCzech/TransNetV2)