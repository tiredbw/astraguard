"""Дообучение детектора (Qwen 3.5 4B) в Google Colab на T4 через Unsloth: LoRA в 16 битах, только текст.

В Colab загрузить в /content три файла: colab_train.py, train.jsonl, valid.jsonl
(train/valid собирает backend/train/sft_build.py в data/sft/). Затем в ячейках:

    !pip install -q unsloth
    !python colab_train.py train      # обучение, адаптер в /content/lora
    !python colab_train.py gguf       # слияние и экспорт GGUF q4_k_m в /content/gguf

Режим рассуждений выключен (как think:false в сервисе), учим только ответ ассистента.
"""

import json
import os
import sys

MODEL = "unsloth/Qwen3.5-4B"
MAX_LEN = 2048


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(s) for s in fh.read().split("\n") if s.strip()]


def text_tokenizer(tok):
    return getattr(tok, "tokenizer", tok)  # Для мультимодальной модели приходит процессор.


def to_text(tok, rows):
    from datasets import Dataset
    texts = [tok.apply_chat_template(r["messages"], tokenize=False, enable_thinking=False) for r in rows]
    return Dataset.from_dict({"text": texts})


def train():
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=MAX_LEN, load_in_4bit=False, load_in_16bit=True)
    tok = text_tokenizer(tok)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42)

    train_ds, valid_ds = to_text(tok, load("train.jsonl")), to_text(tok, load("valid.jsonl"))
    print("пример:\n", train_ds[0]["text"][-600:])
    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=train_ds, eval_dataset=valid_ds,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=MAX_LEN, packing=False,
            per_device_train_batch_size=2, gradient_accumulation_steps=4, num_train_epochs=1,
            max_steps=int(os.environ.get("MAX_STEPS", "-1")),  # Для пробного прогона: MAX_STEPS=10.
            learning_rate=2e-4, warmup_steps=10, lr_scheduler_type="linear", optim="adamw_8bit",
            weight_decay=0.01, fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            logging_steps=10, eval_strategy="no", save_strategy="no", output_dir="outputs",
            report_to="none", seed=42))
    # Потери считаем только по ответу ассистента: системный промпт и запрос модель не учит.
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")
    trainer.train()
    print("валидация:", trainer.evaluate())
    model.save_pretrained("lora")
    tok.save_pretrained("lora")
    print("адаптер сохранён в /content/lora")


def gguf():
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained("lora", max_seq_length=MAX_LEN, load_in_4bit=False, load_in_16bit=True)
    model.save_pretrained_gguf("gguf", text_tokenizer(tok), quantization_method="q4_k_m")
    print("GGUF сохранён в /content/gguf")


if __name__ == "__main__":
    {"train": train, "gguf": gguf}[sys.argv[1]]()
