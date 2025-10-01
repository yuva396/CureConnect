FROM python:3.13.2

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["chainlit", "run", "model.py", "--host", "0.0.0.0", "--port", "7860"]
