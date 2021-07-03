FROM python:3.8-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./ConsumerComplainModel.pkl /deploy/
COPY ./ConsumerComplainTfidfVectorizer.pkl /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
ENTRYPOINT ["python"] 
CMD ["app.py"]
