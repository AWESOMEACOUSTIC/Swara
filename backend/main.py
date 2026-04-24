import base64
from typing import List
import uuid
import modal
import os
import boto3

app = modal.App("swara")
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step", "cd /tmp/ACE-Step && pip install ."])
    .env({"HF_HOME": "/.cache/huggingface"})
    .add_local_python_source("prompts")
)


modal_volume = modal.Volume.from_name("ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

swara_gen_secrets = modal.Secret.from_name("swara-gen-secret")

@app.cls(
    image=image,
    gpu = "Nvidia L4",
    volumes={"/models": modal_volume, "/.cache/huggingface": hf_volume},
    secret=[swara_gen_secrets],
    scaledown_window=15,
)

class MusicGenServer:
    @modal.enter()
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Music Generation Model
        self.music_model = ACEStepPipeline(
            checkpoint_dir =  "/models",
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False
        )

        # Large Language Model
        model_id = "Qwen/Qwen2-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface"
        )

    


@app.local_entrypoint()
def main():
    function_test.remote()