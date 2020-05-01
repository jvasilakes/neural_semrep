FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /usr/src/app
RUN apt-get update --yes && \
	apt-get install --yes less vim
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt

COPY data/ ./data/
COPY bert_model.py  bert_model.py
COPY bert_tokenization.py bert_tokenization.py
COPY predication_filter.py predication_filter.py
COPY relationship_extraction.py relationship_extraction.py
COPY experiment_scripts/ ./experiment_scripts/
