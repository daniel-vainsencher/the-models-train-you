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
base_path = "/DeepSpeedExamples/applications/DeepSpeed-Chat/training"

@stub.function(gpu="A100",
               timeout=86400,
               secret=modal.Secret.from_name("huggingface-secret"))
def do_sft(num_train_epochs, max_total_steps, model_name, target_model_name):
    hf_token = os.environ["HUGGINGFACE_TOKEN"]
    cmds = [
            "date",
            "git config --global credential.helper store",
            f"huggingface-cli login --token {hf_token} --add-to-git-credential",
            f"cd {base_path} && git pull",
            "date",
            f"""export TOKENIZERS_PARALLELISM=false && \
            cd {base_path}/step1_supervised_finetuning/ && \
            /usr/bin/python -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMF19 \
            --master_addr=127.0.0.1 --master_port=29500 --enable_each_rank_log=None \
            main.py --model_name_or_path {model_name} --target_model_name {target_model_name} \
            --data_split 2,4,4 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --max_seq_len 512 \
            --lora_dim 128 --only_optimize_lora \
            --max_total_steps {max_total_steps} \
            --learning_rate 9.65e-6 --weight_decay 0. --num_train_epochs {num_train_epochs} --lr_scheduler_type cosine \
            --num_warmup_steps 0 --gradient_accumulation_steps 1 \
            --zero_stage 2 --deepspeed \
            --output_dir /DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/1.3b \
            --data_path danielv835/personal_finance_v0.2 Dahoas/rm-static Dahoas/full-hh-rlhf \
            Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets stanfordnlp/SHP""",
            ]
    #--gradient_checkpoint \ 
    for cmd in cmds:
        print(f"About to run: \n{cmd}\n")
        subprocess.run(cmd, shell=True)

parameter_presets = {
    "lightest": {
        "num_train_epochs": 1,
        "target_model_name": "danielv835/PF_Coach_sft_lightest",
        "model_name": "facebook/opt-350m",
        "max_total_steps": 5,
        }
    }

parameter_defaults = {
    "num_train_epochs": 2,
    "target_model_name": "danielv835/PF_Coach_sft_1.3b",
    "model_name": "facebook/opt-1.3b",
    "max_total_steps": 1000,
}

@stub.local_entrypoint()
def cli(preset: str = "lightest"):
    parameters = parameter_defaults | parameter_presets[preset]
    print(f"Apply preset: {preset}, resulting in parameters: {parameters}")
    do_sft.call(**parameters)



