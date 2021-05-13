poseidon
========
poseidon is an ONNX inferencing server based on [ONNX Runtime] and [Flask].

Motivation
----------
We need an ONNX inference server for an academic environment which is able to serve a multitude of models in different versions. Currently, we use tensorflow serving (tf serving) for network-based inferencing of tensorflow saved models. This enables our researchers to access any model in any version over the network without the need to set up a local machine learning environment. tf serving is well suited for this task, as it unloads any unused models and frees up the resources. Currently, there is no inferencing server with the same properties for the ONNX file format.

We expect poseidon to be an interim solution until a more professional software package is available. I.e. [ONNX Runtime server] might be suitable in the future. Currently, it only supports a single model file.


Design Goals
------------
  - Keep it as simple as possible.
    - Minimal code base.
    - HTTP/REST interface.
  - Minimize ressource footprint when not in use.
  - Ability to serve many models in different versions.


Model Directory
---------------
poseidon uses the environment variable `POSEIDON_MODEL_PATH` to set the model base path. If `POSEIDON_MODEL_PATH` is not set poseidon defaults to `./models`. The models must be placed in the model base path according to the following convention:

```
{POSEIDON_MODEL_PATH}/{NAME}/{VERSION}/{FN}
```

For example: `/models/mobilenetv2/7/mobilenetv2-7.onnx`. Only a single ONNX file is allowed per `{NAME}/{VERSION}`.


Docker
------
You may deploy poseidon using Docker. Copy or mount the model base path into the docker environment and start the server:

```bash
docker run --rm \
  -p 80:80 \
  hbwinther/poseidon
```

You may deploy your own model directory by mounting your local model base path into the docker environment: `-v "/MYMODELDIR:/models"`


API Endpoints
-------------
  - /list
  - /model/{NAME}/{VERSION}:info
  - /model/{NAME}/{VERSION}:inference


[ONNX Runtime]: https://www.onnxruntime.ai
[Flask]: https://flask.palletsprojects.com
[ONNX Runtime server]: https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Server_Usage.md