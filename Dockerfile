FROM python:3.9

WORKDIR /app/

COPY ./src/ ./app
COPY ./requirements.txt ./


# 必要なモジュールをインストール
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "bash" ]