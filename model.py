import pyaudio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import wave
import soundfile as sf
import os
import librosa

model_path = "/home/dark/AssistModel/whisper_finetuned/final_model"  # Update to your model folder, e.g., "C:/Users/YourUsername/Project/whisper_base_finetuned"

# Verify model path exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path '{model_path}' does not exist. Ensure 'whisper_base_finetuned' folder is in the correct directory.")

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5 # Adjustable: try 3-7 seconds for better context

# Load the fine-tuned processor and model
try:
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
except Exception as e:
    raise Exception(f"Error loading model from {model_path}: {str(e)}")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")


# Real-time recognition function
def recognize_speech( record_seconds=RECORD_SECONDS , start = "" ):
    
    if start == "jasfer":
        audio = pyaudio.PyAudio()
        try:
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        except Exception as e:
            audio.terminate()
            raise Exception(f"Error opening microphone stream: {str(e)}")

        print(f"Recording for {record_seconds} seconds... Speak clearly!")
        frames = []

        for _ in range(0, int(RATE / CHUNK * record_seconds)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        print("Finished recording.")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save to temporary .wav file
        temp_audio = "temp_audio.wav"
        wf = wave.open(temp_audio, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Process audio with noise reduction and normalization
        try:
            # Load audio
            audio_data, sr = librosa.load(temp_audio, sr=RATE)
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            # Basic noise reduction (trim silence)
            audio_data, _ = librosa.effects.trim(audio_data, top_db=20)

            # Process for Whisper
            inputs = processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=int(30 * 16000)
            ).input_features.to(device)

            # Generate transcription
            predicted_ids = model.generate(
                inputs,
                max_length=448,
                num_beams=4,  # Beam search for better accuracy
                no_repeat_ngram_size=2  # Prevent repetitive outputs
            )
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            print(f"Transcription: {transcription}")
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
        finally:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)  # Clean up temporary file
    
    return transcription

# # Run real-time recognition in a loop
while True:
    try:
        recognize_speech(start = "jasfer")
        user_input = input("Continue? (y/n): ").lower()
        if user_input != 'y':
            break
    except Exception as e:
        print(f"Error in recognition loop: {str(e)}")
        break