#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path, environ
import re
from glob import glob
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from flask_json import FlaskJSON, JsonError, json_response, as_json
import numpy as np
import onnxruntime as ort

app = Flask(__name__)
app.config['MODEL_PATH'] = environ.get('POSEIDON_MODEL_PATH', default='./models')
app.config['JSON_ADD_STATUS'] = False
FlaskJSON(app)

model_name_pattern = '.*\/(?P<model_name>.*)\/(?P<model_version>.*)\/(?P<model_fn>.*\.onnx)'
model_name_extractor = re.compile(model_name_pattern)


def get_model(name, version):
    # find corresponding model file (exactly ONE onnx file)
    candidates = glob(path.join(app.config['MODEL_PATH'], name, version, '*.onnx'))
    if len(candidates) == 0:
        raise BadRequest('no suitable model found')
    if len(candidates) > 1:
        raise BadRequest('ambiguous model definition')
    fn = candidates[0]

    # load onnx file
    ort_session = ort.InferenceSession(fn)
    return ort_session


def dtype_orrt2np(dtype):
    # ToDo:
    # need to find complete list
    if dtype == 'tensor(float)':
        return np.float32
    else:
        raise NotImplementedError()


@app.route('/')
def hello():
    return 'poseidon inference server\n'


@app.route('/list', methods=['GET'])
@as_json
def list_models():
    models = {}
    for fn in glob(path.join(app.config['MODEL_PATH'], '*', '*', '*.onnx')):
        m = model_name_extractor.match(fn)
        if not m: # not a valid naming scheme
            continue
        m = m.groupdict()
        if not m['model_name'] in models:
            models[m['model_name']] = []
        models[m['model_name']].append(m['model_version'])
    return {'models': models}


@app.route('/model/<name>/<version>:info', methods=['GET'])
@as_json
def model_info(name, version):
    # load onnx file
    ort_session = get_model(name, version)
    metadata = ort_session.get_modelmeta()

    info = {}
    for key, value in metadata.custom_metadata_map.items():
        info[key] = value
        
    for item in ['description', 'domain', 'graph_description', 'graph_name',
                 'producer_name', 'version']:
        info[item] = getattr(metadata, item)

    info['inputs'] = []
    for input in ort_session.get_inputs():
        v = {'name': input.name,
             'shape': input.shape,
             'type': input.type}
        info['inputs'].append(v)
    
    info['outputs'] = []
    for output in ort_session.get_outputs():
        v = {'name': output.name,
             'shape': output.shape,
             'type': output.type}
        info['outputs'].append(v)

    return info


@app.route('/model/<name>/<version>:inference', methods=['POST'])
@as_json
def model_inference(name, version):
    # load onnx file
    ort_session = get_model(name, version)
    metadata = ort_session.get_modelmeta()

    # load input data
    data = request.json
    if (data is None) or ('inputs' not in data):
        raise BadRequest('input data required')
    
    input_dtypes = {}
    for input in ort_session.get_inputs():
        input_dtypes[input.name] = dtype_orrt2np(input.type)
    inputs = {}
    for key, value in data['inputs'].items():
        if not key in input_dtypes:
            raise BadRequest(f'input "{key}" not found in model')
        inputs[key] = np.array(data['inputs'][key], dtype=input_dtypes[key])
    
    # get list of exptected output nodes
    outputs = [output.name for output in ort_session.get_outputs()]
    
    # run inference
    results = ort_session.run(outputs, inputs)
    
    rv = {}
    for output, result in zip(outputs, results):
        rv[output] = result.tolist()
    
    return rv


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
