# CoachLM <img style="float: right;" src="asset/coachLM.png" width="60">

This repo contains codes and the human-created training dataset for CoachLM, an automatic instruction revision approach for LLM instruction tuning.
## 📣 Introduction
Instruction tuning is crucial for enabling Language Learning Models (LLMs) in responding to human instructions. The quality of instruction pairs used for tuning greatly affects the performance of LLMs. However, the manual creation of high-quality instruction datasets is costly, leading to the adoption of automatic generation of instruction pairs by LLMs as a popular alternative. To ensure the high quality of LLM-generated instruction datasets, several approaches have been proposed. Nevertheless, existing methods either compromise dataset integrity by filtering a large proportion of samples, or are unsuitable for industrial applications. Instead of discarding low-quality samples, we propose CoachLM, a novel approach to enhance the quality of instruction datasets through automatic revisions on samples in the dataset. CoachLM is trained from the samples revised by human experts and significantly increases the proportion of high-quality samples in the dataset from 17.7% to 78.9%. The effectiveness of CoachLM is further assessed on various real-world instruction test sets. The results show that CoachLM improves the instruction-following capabilities of the instruction-tuned LLM by an average of 29.9%, which even surpasses larger LLMs with nearly twice the number of parameters.
<p align="center">
    <img src="asset/illustration.png" alt="logo" width=300 height=300 />
</p>
## 🔰 Installation
```
$ pip install requirements.txt
```
## ✨ Expert Revision Dataset
We created a dataset of 2301 samples containing the raw instruction pairs from the Alpaca52k dataset and the human-revised results in order to improve the quality of these LLM-generated instruction pairs.
```
Expert Revision Dataset.json
{
  "Raw Instruction": "",
  "Raw Input": "",
  "Raw Response": "",
  "Revised Instruction": "",
  "Revised Input": "",
  "Revised Response": "",
  "Distance": ""
}
```
As inllustrated above, the dataset contain raw instructions and the revised versions. it also records the edit distances between the raw instructions and the revised samples.
## 📝 Training CoachLM
CoachLM in this repository is implemented by fine-tuning ChatGLM2 on our curated expert-revision dataset. Thanks [ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning) for implementing an efficient tool to fine-tune ChatGLM2. The training steps of CoachLM are as follows:


(1) Determine the subset used for training


Training CoachLM using the whole dataset may not lead to optimal results. We recommend using around the first 30% samples in the dataset sorted by edit distance from largest to smallest. Other approaches of selecting the dataset are also possible.


(2) Format the training dataset


Format the training dataset according to data/example.json, and register the dataset in data/dataset_info.json.  
```
example.json
{
    "instruction": "Improve the following instruction, input and response pair to be more specific, detailed with more logical steps and grammarly corrected.",
    "input": "Instruction: Name three natural elements. Response: Water, air, and fire.",
    "output": "Instruction: Name three natural elements. Response: Some examples of natural elements are:\n\n- Oxygen: This is the most abundant element in the Earth's crust and the second most abundant element in the atmosphere. Oxygen is essential for life, as it is involved in cellular respiration and other metabolic processes. \n\n- Iron: This is the most abundant element in the Earth's core and the fourth most abundant element in the Earth's crust. Iron is a metal that has many physical and chemical properties, such as strength, magnetism, and conductivity. Iron is also important for life, as it is a component of hemoglobin, which transports oxygen in the blood. \n\n- Gold: This is one of the rarest and most valuable elements on Earth. Gold is a metal that has a shiny yellow color and a high density. Gold is resistant to corrosion and oxidation, which makes it suitable for jewelry and coins."
}
```
```
dataset_info.json
{
  "example": {
    "file_name": "example.json",
    "file_sha1": "",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

(3) Start training
```
python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_your_chatglm2_model \
    --do_train \
    --dataset name_of_your_dataset_in_dataset_info_json \
    --finetuning_type lora \
    --output_dir path_to_CoachLM_checkpoint \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --save_strategy "epoch" \
    --learning_rate 2e-4 \
    --num_train_epochs 7 \
    --fp16 true \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_target "query_key_value,dense,dense_h_to_4h,dense_4h_to_h" \
    --use_v2 true \
```

(4) Inference


The inference dataset should be formatted the same as example.json, with output field empty. 
```
python src/train_bash.py \
    --stage sft \
    --do_predict \
    --finetuning_type lora \
    --dataset dataset_for_inference \
    --model_name_or_path path_to_your_chatglm2_model \
    --checkpoint_dir path_to_CoachLM_checkpoint \
    --output_dir path_to_inference_result \
    --per_device_eval_batch_size 32 \
    --predict_with_generate  \
    --eval_num_beams 1  \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_target "query_key_value,dense,dense_h_to_4h,dense_4h_to_h" \
    --use_v2 true \
```
For more information, please refer to [ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning).

## 🧪 CoachLM150 Test Set
We created a instruction-following test suite for LLMs, containing 150 questions covering topics from information extraction, scientific inference, dialogue completion, brainstorming, in-domain question answering, and more. For each question, a reference response is provided by human.
```
CoachLM150 Test Set.json
{
  "instruction": "",
  "input": "",
  "reference response": ""
}
```
