
FROM python:3.10-slim
RUN mkdir server 

WORKDIR /server

RUN python3 -m pip install --upgrade pip
COPY requirements.txt run_server.sh main.py test_server.py ./ 
RUN pip install -r requirements.txt

ENV PATH_TO_TRANSFORMER="transformer.pkl"
ENV PATH_TO_MODEL="model.pkl"
ENV PORT="8000"


CMD ["bash", "run_server.sh"] 
