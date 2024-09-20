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
def record_audio(filename, duration=5, fs=16000):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=3)
    sd.wait() # wait for recording to finish
    sf.write(filename, recording, fs) # save recording as file
    print("Recording complete!")

# function to convert speech (audio file) to text using OpenAIs Whisper API
def speech_to_text(audio_file_path):
    audio_file = open(audio_file_path, "rb")
    transcription = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    #print("Transcribed query from recording: " + transcription.text)
    print(transcription.text)
    return transcription.text
    
# function to get a response from gpt4o
def get_gpt_response(transcribed_query):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers queries in 2-3 sentences."},
            {"role": "user", "content": transcribed_query}
        ]
    )
    #print(response.choices[0].message.content)
    return response.choices[0].message.content

def gpt_response_to_sound_file(gpt_response_text):
    speech_file_path = "output_sound_file_to_user.mp3"
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=gpt_response_text
    )

    with open(speech_file_path, 'wb') as audio_file:
        audio_file.write(response.content)
    
def play_sound_file(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait until the sound finishes playing
        continue
    pygame.mixer.quit()

if __name__ == "__main__":
    record_audio("my_recording.wav")
    transcribed_query = speech_to_text("my_recording.wav") #transcribe audio
    response_to_query = get_gpt_response(transcribed_query) #get response from gpt from transcription

    gpt_response_to_sound_file(response_to_query) #turn gpt response to sound file
    play_sound_file("output_sound_file_to_user.mp3")
