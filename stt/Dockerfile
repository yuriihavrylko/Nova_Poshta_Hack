FROM python:3.9 as builder2
WORKDIR /usr/src/app
RUN mkdir -p /usr/src/app/audio
RUN mkdir -p /usr/src/app/models

RUN wget https://huggingface.co/theodotus/stt_uk_squeezeformer_ctc_ml/resolve/main/stt_uk_squeezeformer_ctc_ml.nemo -P /usr/src/app/models
RUN wget https://huggingface.co/nvidia/stt_en_fastconformer_transducer_large/resolve/main/stt_en_fastconformer_transducer_large.nemo -P /usr/src/app/models


FROM builder2
WORKDIR /usr/src/app
COPY requirements.txt requirements.txt
RUN pip install cython setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py app.py
ENV FLASK_APP=app

CMD [ "python", "app.py" ]
EXPOSE 8890

