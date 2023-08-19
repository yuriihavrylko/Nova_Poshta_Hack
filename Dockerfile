FROM python:3.11

USER root

WORKDIR /app
COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . /app/

EXPOSE 8888

ENTRYPOINT jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
