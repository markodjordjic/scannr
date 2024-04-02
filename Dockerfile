FROM python:3.11

COPY scannr/. /opt/app/scannr
COPY tests/. /opt/app/tests

COPY requirements.txt /opt/app/

WORKDIR /opt/app
RUN pip install -r requirements.txt


