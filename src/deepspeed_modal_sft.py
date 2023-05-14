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
def do_sft():
    hf_token = os.environ["HUGGINGFACE_TOKEN"]
    cmds = ["git clone https://github.com/daniel-vainsencher/DeepSpeedExamples.git -b pf_coach",
            "cd DeepSpeedExamples/applications/DeepSpeed-Chat/ && pip install -r requirements.txt",
            "git config --global credential.helper store",
            f"huggingface-cli login --token {hf_token} --add-to-git-credential",
            """cd DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/ && \
            /usr/bin/python -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMF19 \
            --master_addr=127.0.0.1 --master_port=29500 --enable_each_rank_log=None \
            main.py --model_name_or_path facebook/opt-1.3b --target_model_name danielv835/PF_Coach_sft_1.3b \
            --data_split 2,4,4 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --max_seq_len 512 \
            --learning_rate 9.65e-6 --weight_decay 0. --num_train_epochs 3 --lr_scheduler_type cosine \
            --num_warmup_steps 0 --gradient_accumulation_steps 1 --gradient_checkpoint \
            --zero_stage 2 --deepspeed \
            --output_dir /DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/1.3b \
            --data_path danielv835/personal_finance_v0.2 Dahoas/rm-static Dahoas/full-hh-rlhf \
            Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets stanfordnlp/SHP""",
            ]
    for cmd in cmds:
        subprocess.run(cmd, shell=True)

@stub.local_entrypoint()
def main():
    do_sft.call()



