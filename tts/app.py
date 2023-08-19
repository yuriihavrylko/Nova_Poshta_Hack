from flask import Flask, request, send_file
import wave
import uuid
from balacoon_tts import TTS

app = Flask(__name__)


models = {
    "en": TTS("models/en_us_cmuartic_jets_cpu.addon"),
    "uk": TTS("models/uk_ltm_jets_cpu.addon"),
}


@app.route("/synthesize", methods=["POST"])
def api():
    language = request.json.get("language", "uk")
    if language not in models:
        return "Unsupported language", 400

    if "text" not in request.json:
        return "Missing text", 400

    text = request.json.get("text")

    filename = synthesize(text, language)
    return send_file(f"audio/{filename}", mimetype="audio/wav")


def synthesize(text, language):
    id = uuid.uuid4().hex

    tts = models[language]
    supported_speakers = tts.get_speakers()
    samples = tts.synthesize(text, supported_speakers[-1])

    with wave.open(f"audio/{id}.wav", "w") as fp:
        fp.setparams((1, 2, tts.get_sampling_rate(), len(samples), "NONE", "NONE"))
        fp.writeframes(samples)
    return f"{id}.wav"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8889)
