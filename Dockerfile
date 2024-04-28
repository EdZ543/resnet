FROM python:3.12

WORKDIR /

COPY . /

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "train.py"]