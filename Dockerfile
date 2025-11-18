FROM python:3.9

WORKDIR /code

COPY requirements.txt /code/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /code/

EXPOSE 7860

CMD ["gunicorn", "--workers", "1", "--timeout", "0", "--bind", "0.0.0.0:7860", "app:app"]