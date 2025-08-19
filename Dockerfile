FROM continuumio/miniconda3

WORKDIR /app

COPY ./conda_environment.yml /app/conda_environment.yml

RUN conda env create -f conda_environment.yml --quiet

SHELL ["conda", "run", "-n", "housing", "/bin/bash", "-c"]

COPY ./app /app/app
COPY ./model /app/model
COPY ./data /app/data

EXPOSE 8000

CMD ["conda", "run", "-n", "housing", "gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]

