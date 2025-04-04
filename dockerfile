FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN pip install --no-cache-dir \
    tensorboardX==2.6.2.2 \
    mamba-ssm==2.2.4 \
    casual_conv1d \
    timm==1.0.9 \
    einops==0.8.1 \
    transformers==4.50.0 \
    Pillow==11.1.0 \
    requests==2.32.3 \
    opencv-python \
    scipy \
    flask \
    python-dotenv \
    pymongo \
    matplotlib \
    scikit-image \
    scikit-learn \
    wandb \
    elasticsearch \
    seaborn \
    albumentations \
    fightingcv-attention \
    positional-encodings[pytorch,tensorflow] \
    pytorch_lightning \
    mediapipe \
    datasets

