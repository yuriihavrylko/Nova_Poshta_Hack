import requests
import wavfile
from pydub import AudioSegment

TTS_SERVER = "http://tts.local:8889/synthesize"
STT_SERVER = "http://stt.local:8890/transcribe"

def tts(text_to_voice, language):
    response = requests.post(
        TTS_SERVER,
        json={"text": text_to_voice, "language": language},
        stream=True,
    )
    return response.raw.read()


def stt(audio_bytes, language):
    with open("track.wav", "wb") as fp:
        fp.write(audio_bytes)

    sound = AudioSegment.from_wav("track.wav")
    sound = sound.set_channels(1)
    sound.export("mono_track.wav", format="wav")

    response = requests.post(
        f"{STT_SERVER}?language={language}",
        files={"file": open("mono_track.wav", "rb")},
    )
    return response.json()["transcription"]
