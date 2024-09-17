import sounddevice as sd
import soundfile as sf
import openai
import pygame
from pathlib import Path
from dotenv import load_dotenv  # for loading environment variables
import os  # for accessing environment variables

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# function to record audio from microphone
def record_audio(filename, duration=7, fs=16000):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait() # wait for recording to finish
    sf.write(filename, recording, fs) # save recording as file
    print("Recording complete!")

# function to convert speech (audio file) to text using OpenAIs Whisper API
def speech_to_text(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file: # this line may fail because its not exactly whats in the docs
        transcription = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcription['text']
    
# function to get a response from gpt4o
def get_gpt_response(transcribed_text):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": transcribed_text}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    sample_text = "Say something cool!"
    print(get_gpt_response(sample_text))
