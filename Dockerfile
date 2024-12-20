FROM python:3.13.1
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt
COPY ./insurance_data.db /code/
COPY ./model_generator.py /code/model_generator.py
COPY ./app /code/app
RUN python model_generator.py
RUN mv random_forest_pipeline.pkl app/random_forest_pipeline.pkl
RUN rm /code/insurance_data.db
CMD ["fastapi", "run", "app/main.py", "--port", "80"]