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
)

stub = modal.Stub("gpu_check", image=dschat_image)

@stub.function(gpu="A100",
               timeout=86400,
               secret=modal.Secret.from_name("huggingface-secret"))
def train_rm():
    hf_token = os.environ["HUGGINGFACE_TOKEN"]
    cmds = ["git clone https://github.com/daniel-vainsencher/DeepSpeedExamples.git -b pf_coach",
            "cd DeepSpeedExamples/applications/DeepSpeed-Chat/ && pip install -r requirements.txt",
            "git config --global credential.helper store",
            f"huggingface-cli login --token {hf_token} --add-to-git-credential",
            """cd training/step2_reward_model_finetuning/ && \
            deepspeed --num_gpus 1 \
            main.py --model_name_or_path facebook/opt-350m --target_model_name danielv835/PF_Critic_350m \
            --num_padding_at_beginning 1 --weight_decay 0.1 --disable_dropout \
            --gradient_accumulation_steps 4 --zero_stage 2 \
            --data_path danielv835/personal_finance_v0.2 Dahoas/rm-static Dahoas/full-hh-rlhf \
            Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets stanfordnlp/SHP \
            --gradient_checkpoint --deepspeed --output_dir OUTPUT | tee training.log""",
            ]
    for cmd in cmds:
        subprocess.run(cmd, shell=True)

@stub.local_entrypoint()
def main():
    train_rm.call()


    
