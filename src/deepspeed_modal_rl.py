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
def train_bot():
    hf_token = os.environ["HUGGINGFACE_TOKEN"]
    cmds = ["git clone https://github.com/daniel-vainsencher/DeepSpeedExamples.git -b pf_coach",
            "cd DeepSpeedExamples/applications/DeepSpeed-Chat/ && pip install -r requirements.txt",
            "git config --global credential.helper store",
            f"huggingface-cli login --token {hf_token} --add-to-git-credential",
            """cd training/step3_rlhf_finetuning/ && \
            deepspeed --num_gpus 1 main.py \
                --actor_model_name_or_path danielv835/PF_Coach_sft_1.3b \
                --critic_model_name_or_path danielv835/PF_Critic_350m \
                --target_actor_model_name danielv835/PF_Coach_bot_1.3b \
                --actor_zero_stage 0 --critic_zero_stage 0 \
                --num_padding_at_beginning 1 --gradient_accumulation_steps 2 \
                --deepspeed --actor_lora_dim 128 --enable_hybrid_engine --actor_gradient_checkpointing --disable_actor_dropout \
                --critic_gradient_checkpointing --offload_reference_model \
                --output_dir output | tee training.log""",
            ]
    for cmd in cmds:
        subprocess.run(cmd, shell=True)

@stub.local_entrypoint()
def main():
    train_bot.call()

# If the PF specific models do not exist yet, can use instead:
#                --actor_model_name_or_path facebook/opt-1.3b \
#                --critic_model_name_or_path facebook/opt-350m \
