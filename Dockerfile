FROM nvcr.io/nvidia/pytorch:23.10-py3

# Force stable NumPy and recent Transformers to fix the ndarray TypeError
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    "transformers>=4.45.0" \
    unsloth \
    trl \
    peft \
    accelerate \
    bitsandbytes \
    datasets \
    wandb \
    hf_transfer \
    soxr

# Enable faster downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app
COPY hf_space_trainer.py .

# Increase shared memory for Unsloth
ENV SHM_SIZE=2G

# Set the entry point
CMD ["python", "hf_space_trainer.py"]
