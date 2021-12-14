FROM python:3.8.12-slim-buster

RUN pip install pipenv


WORKDIR /app


COPY ["requirements.txt", "./"]


RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt


COPY ["predict.py", "model.bin", "./"]

EXPOSE 9696 


ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]