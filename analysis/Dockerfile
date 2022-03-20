
FROM jupyter/base-notebook:lab-3.3.2

USER root
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends build-essential python-dev

USER ${NB_UID}
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .
