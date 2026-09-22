FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Runtime libraries required by binary geospatial wheels and common CRS/data IO.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE /app/
COPY native /app/native
COPY src /app/src

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install .

ENTRYPOINT ["fastgc"]

