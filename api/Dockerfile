FROM python:3.13.0 AS base


ENV PYTHONPATH=$PYTHONPATH:/src/

COPY ./requirements.txt requirements.txt

WORKDIR /src

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./ /src/

FROM base AS test
RUN ["python", "-m", "pytest", "-v", "/src/app/tests"]

FROM base AS build

COPY . .
EXPOSE 8080
CMD ["python3", "app/app.py",  "--ip=0.0.0.0", "--port=8080", "--no-browser", "--allow-root"]
