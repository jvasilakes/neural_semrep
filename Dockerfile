FROM tensorflow/tensorflow:nightly-gpu-py3

WORKDIR /usr/src/app
RUN apt-get update --yes && \
	apt-get install --yes less vim
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt

COPY data/ ./data/
COPY bert_models.py bert_models.py
COPY bert_tokenization.py bert_tokenization.py
COPY utils.py utils.py
COPY relationship_classification.py relationship_classification.py
COPY run_prediction.py run_prediction.py
COPY experiment_scripts/ ./experiment_scripts/
