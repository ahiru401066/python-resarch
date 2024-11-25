FROM python:3.9

WORKDIR /app/

COPY ./app/ ./app


# 必要なモジュールをインストール
RUN pip install --no-cache-dir -r ./app/requirements.txt

CMD [ "bash" ]