import requests

TTS_SERVER = "http://localhost:8889/synthesize"
STT_SERVER = "http://localhost:9000/asr?task=transcribe&encode=true&output=txt"

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
    response = requests.post(
        f"{STT_SERVER}&language={language}",
        files={"audio_file": open("track.wav", "rb")},
    )
    return response.text
