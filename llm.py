from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from modelscope.hub.snapshot_download import snapshot_download

def load_llama(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model


def load_llm_local(local_model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    return tokenizer, model

def generate_answer(tokenizer, model, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,   # deterministic
        top_p=1.0,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def generate_batch(tokenizer, model, prompts, max_new_tokens=256):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_p=1.0,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    results = []

    for i in range(len(prompts)):
        # 每个样本真实长度（不包括padding）
        input_len = attention_mask[i].sum().item()

        # 截取生成部分
        generated_tokens = outputs[i][input_len:]

        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        results.append(text)

    return results


if __name__ == "__main__":
    # Download model to local directory
    cache_dir = "./models" # 指定本地目录
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # model_dir = snapshot_download(
    #     model_id="LLM-Research/Meta-Llama-3.1-8B-Instruct",
    #     cache_dir=cache_dir
    # )

    # print("Model downloaded to:", model_dir)

    model_dir = snapshot_download(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        cache_dir=cache_dir
    )

    print("Model downloaded to:", model_dir)