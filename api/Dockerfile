FROM python:3.9.0 AS base


ENV PYTHONPATH=$PYTHONPATH:/app/

COPY ./requirements.txt requirements.txt

WORKDIR /app

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./ /app/

#FROM base AS test
#RUN ["python", "-m", "pytest", "-v", "/src/app/tests"]

FROM base AS build

COPY . .
EXPOSE 8080

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]
