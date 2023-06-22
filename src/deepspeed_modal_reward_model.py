import modal
import subprocess
import os

from deepspeed_modal_shared import stub, base_path, dschat_image, initial_dse_commands

@stub.function(gpu="A100",
               timeout=86400,
               secret=modal.Secret.from_name("huggingface-secret"))
def train_rm():
    hf_token = os.environ["HUGGINGFACE_TOKEN"]
    #assert False, "TODO: take the pre-existing HF reward model name, because the DS abomination cannot be pushed."
    cmds = initial_dse_commands(hf_token) + [
        f"""export TOKENIZERS_PARALLELISM=false && \
        cd training/step2_reward_model_finetuning/ && \
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
