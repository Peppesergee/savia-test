# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /LLM_small
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install build-essential git libgl1 ffmpeg libsm6 libxext6 -y 
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
#CMD ["sleep", "infinity"]
CMD ["python", "flask_client.py"]
