FROM python:3.10

ENV PYTHONUNBUFFERED=1

WORKDIR /code

# Copy requirements early for caching
COPY requirements.txt /code/

# Install TensorFlow 2.20.0 manually
RUN pip install --no-cache-dir tensorflow-cpu==2.20.0

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /code/

EXPOSE 7860

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:7860"]