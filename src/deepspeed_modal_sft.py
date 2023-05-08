import modal
import subprocess

dschat_image = (
    modal.Image.from_dockerhub(
        "nvidia/cuda:11.7.0-devel-ubuntu20.04",
        setup_dockerfile_commands=[
            "RUN apt-get update",
            "RUN apt-get install -y python3 python3-pip python-is-python3",
        ],
    )
    .apt_install("git", "gcc", "build-essential")
    # Create and enter venv in image prep to messing with the gloal env which pip complains about.
    .pip_install("deepspeed[fused_adam]>=0.9.2")
    .run_commands(
        "git clone https://github.com/daniel-vainsencher/DeepSpeedExamples.git",
        "cd DeepSpeedExamples/applications/DeepSpeed-Chat/ && pip install -r requirements.txt",
    )
)

stub = modal.Stub("gpu_check", image=dschat_image)

@stub.function(gpu="A100", timeout=1800)
def tell_me_stuff():
    cmds = ["nvidia-smi -L",
            "uname -a",
            "pip list",
            # "cd /DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/ && /usr/bin/python -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMF19 --master_addr=127.0.0.1 --master_port=29500 --enable_each_rank_log=None main.py --model_name_or_path facebook/opt-1.3b --gradient_accumulation_steps 8 --lora_dim 128 --gradient_checkpoint --zero_stage 0 --deepspeed --output_dir /DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/1.3b",
            ]
    for cmd in cmds:
        print(subprocess.check_output(cmd, shell=True).decode('utf-8').strip())

# "python train.py --actor-model facebook/opt-1.3b --reward-model facebook/opt-350m --deployment-type single_node"

@stub.local_entrypoint()
def main():
    print("Run:\n", tell_me_stuff.call())



