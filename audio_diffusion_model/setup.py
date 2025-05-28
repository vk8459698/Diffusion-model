from setuptools import setup, find_packages

setup(
    name="audio-diffusion-model",
    version="0.1.0",
    description="End-to-End Audio Diffusion Model with Multi-Modal Conditioning",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.20.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "wandb>=0.13.0",
        "matplotlib>=3.5.0",
        "tensorboard>=2.8.0",
        "omegaconf>=2.2.0",
        "hydra-core>=1.2.0",
        "einops>=0.6.0",
    ],
    python_requires=">=3.8",
)
