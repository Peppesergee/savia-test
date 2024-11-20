# syntax=docker/dockerfile:1
#FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM python:3.13.0
WORKDIR /python_client
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install build-essential git libgl1 ffmpeg libsm6 libxext6 -y 
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD ["sleep", "infinity"]