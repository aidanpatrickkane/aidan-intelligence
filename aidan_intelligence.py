import pvporcupine
import pyaudio
import struct
import sounddevice as sd
import soundfile as sf
import openai
from pathlib import Path
from dotenv import load_dotenv  # for loading environment variables
import os  # for accessing environment variables

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

porcupine = pvporcupine.create(
    access_key = os.getenv("ACCESS_KEY"),
    keyword_paths = ['Big-Red_en_raspberry-pi_v3_0_0.ppn']
)

# function to record audio from microphone
def record_audio(filename, duration=7, fs=16000):
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
    os.system(f'mpg123 {file_path}')

if __name__ == "__main__":
    # Initialize PyAudio
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        input_device_index=3,
        frames_per_buffer=porcupine.frame_length
    )

    print("Listening for wake words...")

    try:
        while True:
            # Read audio stream in small chunks and process with Porcupine
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Wake word detected! Listening for your question...")
                
                # Once wake word is detected, run your main logic
                record_audio("my_recording.wav")
                transcribed_query = speech_to_text("my_recording.wav")  # transcribe audio
                response_to_query = get_gpt_response(transcribed_query)  # get response from GPT
                
                gpt_response_to_sound_file(response_to_query)  # generate speech file
                play_sound_file("output_sound_file_to_user.mp3")  # play the speech file
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if porcupine is not None:
            porcupine.delete()
        if audio_stream is not None:
            audio_stream.close()
        if pa is not None:
            pa.terminate()
