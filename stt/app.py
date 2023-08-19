from flask import Flask, request
import wave
import uuid
import nemo.collections.asr as nemo_asr

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "audio"


models = {
    "en": nemo_asr.models.ASRModel.restore_from(
        "models/stt_en_fastconformer_transducer_large.nemo"
    ),
    "uk": nemo_asr.models.ASRModel.restore_from(
        "models/stt_uk_squeezeformer_ctc_ml.nemo"
    ),
}


@app.route("/transcribe", methods=["POST"])
def api():
    language = request.args.to_dict().get("language", "uk")
    if language not in models:
        return "Unsupported language", 400

    if request.method == "POST":
        f = request.files["file"]
        filename = uuid.uuid4().hex + ".wav"
        f.save(filename)
        transcription = transcribe(filename, language)

        return {"transcription": transcription}, 200


def transcribe(file_path, language):
    transcriptions_ua = models[language].transcribe([file_path])
    return transcriptions_ua[0]


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8890)
