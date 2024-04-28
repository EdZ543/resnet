FROM pytorch/pytorch

WORKDIR /

COPY . /

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "train.py"]