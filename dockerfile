FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

# Make sure the environment is activated
SHELL ["conda", "run", "-n", "dsa3101_env", "/bin/bash", "-c"]

COPY . .

CMD ["conda", "run", "-n", "dsa3101_env", "python", "main.py"]