FROM python:3.11.4

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app
COPY ./app/enhanced_student_ssl_dqn_model.pt /code/app/enhanced_student_ssl_dqn_model.pt

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]