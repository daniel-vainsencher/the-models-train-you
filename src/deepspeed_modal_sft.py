import modal
import subprocess
import os

dschat_image = (
    modal.Image.from_dockerhub(
        "nvidia/cuda:11.7.0-devel-ubuntu20.04",
        setup_dockerfile_commands=[
            "RUN apt-get update",
            "RUN apt-get install -y python3 python3-pip python-is-python3 nano hexyl unzip",
        ],
    )
    .apt_install("git", "gcc", "build-essential")
    # Create and enter venv in image prep to messing with the gloal env which pip complains about.
    .pip_install("deepspeed[fused_adam]>=0.9.2", "tensorboard")
    .run_commands(
        "git clone https://github.com/daniel-vainsencher/DeepSpeedExamples.git -b pf_coach",
        "cd DeepSpeedExamples/applications/DeepSpeed-Chat/ && pip install -r requirements.txt",
    )
)

# To do manually in a shell:
# cd /DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/
# /usr/bin/python -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMF19 --master_addr=127.0.0.1 --master_port=29500 --enable_each_rank_log=None main.py --model_name_or_path facebook/opt-1.3b --target_model_name danielv835/PF_Coach_sft --gradient_accumulation_steps 8 --lora_dim 128 --gradient_checkpoint --zero_stage 0 --deepspeed --output_dir /DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/1.3b --data_path danielv835/pf_coach_v0.1 | tee last_training.log

stub = modal.Stub("gpu_check", image=dschat_image)


@stub.function(gpu="A100",
               timeout=86400,
               secret=modal.Secret.from_name("huggingface-secret"))
def tell_me_stuff():
    hf_token = os.environ["HUGGINGFACE_TOKEN"]
    cmds = ["nvidia-smi -L",
            "uname -a",
            "pip list",
            "git config --global credential.helper store",
            f"huggingface-cli login --token {hf_token} --add-to-git-credential",
            "cd /DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/ && /usr/bin/python -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMF19 --master_addr=127.0.0.1 --master_port=29500 --enable_each_rank_log=None main.py --model_name_or_path facebook/opt-1.3b --target_model_name danielv835/PF_Coach_sft --gradient_accumulation_steps 8 --lora_dim 128 --gradient_checkpoint --zero_stage 0 --deepspeed --output_dir /DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/1.3b --data_path danielv835/pf_coach_v0.1 Dahoas/full-hh-rlhf stanfordnlp/SHP",
            ]
    for cmd in cmds:
        subprocess.run(cmd, shell=True)

@stub.local_entrypoint()
def main():
    tell_me_stuff.call()



