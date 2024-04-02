FROM python:3.11

COPY scenario_writing/. /opt/app/scenario_writing
#COPY tests/. /opt/app/tests

#COPY setup.py /opt/app/
COPY requirements.txt /opt/app/

WORKDIR /opt/app
RUN pip install -r requirements.txt

#RUN cd /opt/app/
#RUN python setup.py install

