FROM python:3.10.14-slim

WORKDIR /app

COPY /demo/requirements.txt /app/

RUN pip install -r requirements.txt

COPY ./ /app/

CMD sh run.sh /app/face_detect_api.py
