FROM python:3.11 as builder
WORKDIR /usr/src/app
RUN mkdir -p /usr/src/app/audio
RUN mkdir -p /usr/src/app/models

RUN wget https://huggingface.co/balacoon/tts/resolve/main/uk_ltm_jets_cpu.addon -P /usr/src/app/models
RUN wget https://huggingface.co/balacoon/tts/resolve/main/en_us_cmuartic_jets_cpu.addon -P /usr/src/app/models


FROM builder
WORKDIR /usr/src/app
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py app.py
ENV FLASK_APP=app

CMD [ "python", "app.py" ]
EXPOSE 5000

