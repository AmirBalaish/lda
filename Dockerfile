FROM python:3.5

RUN pip install cython numpy pbr wheel sklearn scipy

VOLUME /lda
WORKDIR /lda

CMD ["./build.sh"]
