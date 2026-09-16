FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vmp/ vmp/
COPY run_experiments.py demo.py ./

CMD ["python", "run_experiments.py"]
