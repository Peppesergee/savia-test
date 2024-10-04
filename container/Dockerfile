FROM ubuntu:24.04
RUN echo "Europe/Rome" > /etc/timezone &&\
apt-get update &&\
apt-get -y install git python3 python3-pip python3-venv &&\
mkdir /rer_venv &&\
cd /rer_venv &&\
python3 -m venv /rer_venv
#mkdir /rer_app &&\
#cd /rer_app/ &&\
#git init &&\
#git remote add origin https://github.com/mantmarco/savia-test.git &&\
#git branch -M main &&\
#git pull origin main
#RUN cd /rer_venv &&\
#. bin/activate &&\
#pip install uvicorn &&\
#pip install fastapi &&\
#pip install transformers &&\
#pip install torch &&\
#pip install bitsandbytes
EXPOSE 8000
