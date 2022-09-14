ARG IMAGE
FROM ${IMAGE}

COPY . .

RUN pip install -e src/dnn

RUN chmod +x ./entrypoint.sh

ENTRYPOINT [ "./entrypoint.sh" ]