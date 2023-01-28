FROM python:3.10
COPY . /app
RUN pip install -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
CMD python /app/main.py