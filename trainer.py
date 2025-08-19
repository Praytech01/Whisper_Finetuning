import os
import pandas as pd
import torch
import numpy as np
import librosa
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Trainer, TrainingArguments
from pathlib import Path

# Setting random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Loading the CSV file with transcriptions
csv_path = "sentences.csv"
data_df = pd.read_csv(csv_path)

# Creating a list of audio file paths
audio_dir = "audio_data"
audio_files = [os.path.join(audio_dir, f"speaker_1_{i}.wav") for i in range(1, len(data_df) + 1)]

# Ensuring the number of audio files matches the number of transcriptions
assert len(audio_files) == len(data_df), "Mismatch between number of audio files and transcriptions"

# Load audio files manually with librosa
def load_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    target_length = 30 * 16000
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    elif len(audio) > target_length:
        audio = audio[:target_length]
    return audio

# Create dataset without Audio feature
dataset = Dataset.from_dict({
    "audio_path": audio_files,
    "sentence": data_df["sentence"].tolist()
})

# Loading Whisper processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Ensuring model is on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def preprocess_function(examples):
    audio_array = load_audio(examples["audio_path"])
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        text=examples["sentence"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=processor.tokenizer.model_max_length
    )
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    if inputs["input_features"].shape[-1] > 3000:
        inputs["input_features"] = inputs["input_features"][:, :3000]
    elif inputs["input_features"].shape[-1] < 3000:
        padding = torch.zeros((inputs["input_features"].shape[0], 3000 - inputs["input_features"].shape[-1]))
        inputs["input_features"] = torch.cat([inputs["input_features"], padding], dim=-1)
    return inputs

# Apply preprocessing
dataset = dataset.map(preprocess_function, remove_columns=["audio_path", "sentence"], num_proc=1)

# Splitting dataset into train and validation
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Data collator
def data_collator(features):
    input_features = [f["input_features"] for f in features]
    labels = [f["labels"] for f in features]
    input_features = [torch.tensor(f, dtype=torch.float32) for f in input_features]
    target_length = 3000
    num_mel_bins = input_features[0].shape[0]
    padded_input_features = torch.zeros(
        (len(input_features), num_mel_bins, target_length),
        dtype=torch.float32
    )
    for i, feature in enumerate(input_features):
        current_length = feature.shape[-1]
        if current_length > target_length:
            padded_input_features[i, :, :target_length] = feature[:, :target_length]
        else:
            padded_input_features[i, :, :current_length] = feature
    max_label_length = max(len(l) for l in labels)
    padded_labels = torch.full(
        (len(labels), max_label_length),
        fill_value=processor.tokenizer.pad_token_id,
        dtype=torch.long
    )
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = torch.tensor(label, dtype=torch.long)
    return {
        "input_features": padded_input_features,
        "labels": padded_labels
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./whisper_finetuned",
    eval_strategy="steps",
    save_strategy="steps",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    push_to_hub=False,
    fp16=torch.cuda.is_available(),
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Start fine-tuning
trainer.train()

# Save model and processor
model.save_pretrained("./whisper_finetuned/final_model")
processor.save_pretrained("./whisper_finetuned/final_model")

# Evaluate
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Test model
# def test_model(audio_path, transcription):
#     audio_array, sr = librosa.load(audio_path, sr=16000)
#     audio = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
#     predicted_ids = model.generate(audio)
#     predicted_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#     print(f"Original: {transcription}")
#     print(f"Predicted: {predicted_text}")

# test_idx = 0
# test_audio = audio_files[test_idx]
# test_transcription = data_df["sentence"].iloc[test_idx]
# test_model(test_audio, test_transcription)