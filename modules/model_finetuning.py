import gc

from datasets import load_dataset
import numpy as np
from peft import LoraConfig, get_peft_model
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class MemoryCleanupCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):  # type: ignore
        # Clears memory after the entire evaluation loop finishes
        gc.collect()
        torch.cuda.empty_cache()

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore
        # If RAM climbs during training too
        if state.global_step % 5 == 0:
            gc.collect()
            gc.collect()
            torch.cuda.empty_cache()


# ==========================================
# 1. Configuration & Model Loading
# ==========================================
model_id = "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0"
MAX_LEN = 4096
max_seq_length = MAX_LEN

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Right padding is safer for standard CausalLM training (non-generation)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Your specific RAG chat template
tokenizer.chat_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ '<start_of_turn>system\n' + messages[0]['content'] + '<end_of_turn>\n' }}"
    "{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}"
    "{{'<start_of_turn>user\n' + message['content'] "
    "+ '<end_of_turn>\n<start_of_turn>model\n' }}"
    "{% elif message['role'] == 'model' %}{{ message['content'] + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
)
model.enable_input_require_grads()

print("Configuring DoRA Adapters...")
peft_config = LoraConfig(
    r=16,  # Cut from 32 to 16
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=False,  # Disabling DoRA is your biggest speed gain
)
model = get_peft_model(model, peft_config)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()


# ==========================================
# 2. Manual Data Masking (The Fix)
# ==========================================
def format_manual_mask(examples):  # type: ignore
    LABELS = ["A", "B", "C", "D", "E", "F"]  # noqa
    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []
    token_type_ids_batch = []
    was_truncated = []

    TOTAL_MAX_TOKENS = MAX_LEN  # noqa

    for ctx, q, opts, ans in zip(
        examples["context"], examples["question"], examples["options"], examples["answer"], strict=False
    ):
        ctx = f"Контекст (уривки з PDF-файлів - кожен уривок розділений з допомогою символів ``` \
            та містить номер сторінки виділений за допомогою []):\n{ctx}\n\n"
        # Updated to request both the letter and the page number
        fixed_text = (
            f"Запитання: {q}\nВаріанти:\n{opts}\nІнструкція:\n- Дай відповідь на Запитання використовуючи Контекст. \
            \n- Поверни літеру правильної відповіді ({' '.join(LABELS)}) \
            та номер сторінки, на якій знайдено інформацію, \
            через пробіл (наприклад: A 1). \
            \n- Добре подумай, спочатку відкинь очевидно нерелевантні відповіді."
        )

        # The exact completion text
        completion_text = f"{ans}<end_of_turn>\n"

        fixed_tokens = tokenizer.encode(fixed_text, add_special_tokens=False)
        ans_tokens = tokenizer.encode(completion_text, add_special_tokens=False)

        context_budget = TOTAL_MAX_TOKENS - len(fixed_tokens) - len(ans_tokens) - 50
        context_budget = max(context_budget, 100)

        ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False)
        original_len = len(ctx_tokens)

        if original_len > context_budget:
            ctx_tokens = ctx_tokens[:context_budget]
            truncated_ctx = tokenizer.decode(ctx_tokens)
            was_truncated.append(True)
        else:
            truncated_ctx = ctx
            was_truncated.append(False)

        user_prompt = f"Контекст:\n{truncated_ctx}\n{fixed_text}"
        prompt_convo = [{"role": "user", "content": user_prompt}]
        prompt_text = tokenizer.apply_chat_template(prompt_convo, tokenize=False)

        # MANUALLY ENCODE AND MASK
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        input_ids = prompt_ids + ans_tokens
        # We explicitly set the prompt to -100 so loss is ONLY calculated on the answer
        labels = [-100] * len(prompt_ids) + ans_tokens
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        input_ids_batch.append(input_ids)
        labels_batch.append(labels)
        attention_mask_batch.append(attention_mask)
        token_type_ids_batch.append(token_type_ids)

    return {
        "input_ids": input_ids_batch,
        "labels": labels_batch,
        "attention_mask": attention_mask_batch,
        "token_type_ids": token_type_ids_batch,
        "was_truncated": was_truncated,
    }


print("Loading and mapping datasets...")
dataset = load_dataset(
    "json", data_files={"train": "../data/train_from_chunks_v2.json", "test": "../data/test_from_chunks_v2.json"}
)

# Map and completely drop the original text columns so Trainer doesn't get confused
dataset = dataset.map(format_manual_mask, batched=True, num_proc=8, remove_columns=dataset["train"].column_names)


# ==========================================
# Print Truncation Statistics
# ==========================================
train_truncation_rate = sum(dataset["train"]["was_truncated"]) / len(dataset["train"])
test_truncation_rate = sum(dataset["test"]["was_truncated"]) / len(dataset["test"])

print("\n--- TRUNCATION STATS ---")
print(f"Train Dataset Truncation Rate: {train_truncation_rate * 100:.2f}%")
print(f"Test Dataset Truncation Rate:  {test_truncation_rate * 100:.2f}%")
print("------------------------\n")


# ==========================================
# 3. Custom Evaluation Metrics
# ==========================================
def preprocess_logits_for_metrics(logits, labels):  # type: ignore
    """
    Condenses the massive logits tensor into predicted token IDs.
    This prevents OOM errors during evaluation.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = logits.argmax(dim=-1)
    return preds


def compute_metrics(eval_preds):  # type: ignore
    preds, labels = eval_preds

    # Standard CausalLM shift
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    batch_size = labels.shape[0]
    ans_correct = 0
    page_correct = 0
    total = 0

    for i in range(batch_size):
        unmasked_indices = np.where(labels[i] != -100)[0]

        if len(unmasked_indices) > 0:
            # Extract the actual and predicted token arrays for the unmasked region
            actual_tokens = labels[i, unmasked_indices]
            pred_tokens = preds[i, unmasked_indices]

            # Decode the arrays into full strings (e.g., "D 1")
            actual_text = tokenizer.decode(actual_tokens, skip_special_tokens=True).strip()
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

            # Split by whitespace to parse the Letter and Page components
            actual_parts = actual_text.split()
            pred_parts = pred_text.split()

            actual_ans = actual_parts[0] if len(actual_parts) > 0 else ""
            actual_page = actual_parts[1] if len(actual_parts) > 1 else ""

            pred_ans = pred_parts[0] if len(pred_parts) > 0 else ""
            pred_page = pred_parts[1] if len(pred_parts) > 1 else ""

            if i == 0:
                print("\n\n--- EVALUATION DEBUG INFO ---", flush=True)
                print(f"Target Label String: '{actual_text}' -> Ans: '{actual_ans}', Page: '{actual_page}'", flush=True)
                print(f"Model Pred String:   '{pred_text}' -> Ans: '{pred_ans}', Page: '{pred_page}'", flush=True)
                print("-----------------------------\n", flush=True)

            # Evaluate Letter and Page separately
            if pred_ans == actual_ans and actual_ans != "":
                ans_correct += 1
            if pred_page == actual_page and actual_page != "":
                page_correct += 1

            total += 1

    return {
        "answer_accuracy": ans_correct / total if total > 0 else 0.0,
        "page_accuracy": page_correct / total if total > 0 else 0.0,
    }


# ==========================================
# 4. Standard Trainer Setup
# ==========================================
training_args = TrainingArguments(
    output_dir="mamay_spark_lora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=1,
    warmup_ratio=0.05,
    num_train_epochs=1,
    learning_rate=1e-5,
    bf16=True,
    tf32=True,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    logging_steps=5,
    gradient_checkpointing=True,  # Saves VRAM at a small compute cost
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Modern implementation
    load_best_model_at_end=True,
    metric_for_best_model="answer_accuracy",  # Tracking the new separated metric
    greater_is_better=True,
    # save_total_limit=1,
    report_to="none",
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

# Using standard Hugging Face Trainer, completely bypassing TRL
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    callbacks=[MemoryCleanupCallback],
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

# ==========================================
# 5. Execute Training & Evaluation
# ==========================================
print("Evaluating raw model performance before training...")
initial_metrics = trainer.evaluate()
gc.collect()
torch.cuda.empty_cache()

print("--- BASELINE METRICS ---")
print(f"Initial Answer Accuracy: {initial_metrics.get('eval_answer_accuracy', 0) * 100:.2f}%")
print(f"Initial Page Accuracy:   {initial_metrics.get('eval_page_accuracy', 0) * 100:.2f}%")
print(f"Initial Loss:            {initial_metrics.get('eval_loss', 0):.4f}")
print("------------------------")

print("Starting Training...")
trainer.train()

print("Saving Best LoRA adapters...")
trainer.model.save_pretrained("mamay_spark_lorafinal")
tokenizer.save_pretrained("mamay_spark_lorafinal")

print("Fine-tuning complete!")
