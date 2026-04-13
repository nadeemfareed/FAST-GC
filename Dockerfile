FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml && conda clean -afy

SHELL ["conda", "run", "-n", "fastgc", "/bin/bash", "-c"]

COPY . /app

RUN pip install -e .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "fastgc", "fastgc"]