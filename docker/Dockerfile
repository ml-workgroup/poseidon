FROM tiangolo/uwsgi-nginx:python3.8

ENV POSEIDON_MODEL_PATH=/models

RUN pip install --upgrade \
        flask flask-restful flask_json \
        onnxruntime==1.7.0
