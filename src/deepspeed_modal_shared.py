import modal
import subprocess
import os

dschat_image = (
    modal.Image.from_dockerhub(
        "nvidia/cuda:11.7.0-devel-ubuntu20.04",
        setup_dockerfile_commands=[
            "RUN apt-get update",
            "RUN apt-get install -y python3 python3-pip python-is-python3 nano hexyl unzip git gcc build-essential git-lfs",
        ],
    )
    # Create and enter venv in image prep to messing with the gloal env which pip complains about.
    .pip_install("deepspeed[fused_adam]>=0.9.2", "tensorboard")
    .run_commands(
        "git clone https://github.com/daniel-vainsencher/DeepSpeedExamples.git -b pf_coach",
        "cd DeepSpeedExamples/applications/DeepSpeed-Chat/ && pip install -r requirements.txt",
        "git lfs install",
        )
)

stub = modal.Stub("DeepSpeedExamples", image=dschat_image)
base_path = "/DeepSpeedExamples/applications/DeepSpeed-Chat"

def initial_dse_commands(hf_token):
    return [
        "date",
        "git config --global credential.helper store",
        f"huggingface-cli login --token {hf_token} --add-to-git-credential",
        f"cd {base_path} && git pull",
        "date",
        f"cd {base_path} && pip install -r requirements.txt",
        "date",
    ]

