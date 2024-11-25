FROM python:3.9

WORKDIR /app/

COPY ./src/ ./app
COPY ./requirements.txt ./

CMD [ "bash" ]