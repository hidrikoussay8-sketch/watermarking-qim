FROM python:3.11-slim

WORKDIR /app

COPY code.py .
COPY image.jpg .

RUN pip install opencv-python-headless numpy matplotlib scipy scikit-image

CMD ["python", "code.py"]