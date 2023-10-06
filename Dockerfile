FROM ubuntu
RUN apt update
RUN apt install python3 python3-pip -y
WORKDIR /app
COPY saved_models ./saved_models
COPY requirements.txt .
COPY inference_pipeline.py .
COPY /steps ./steps
RUN pip3 install -r requirements.txt
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
CMD ["python3","inference_pipeline.py"]


