from project.model.MLP import MLP
params={
    "batch_size": {"_type":"choice", "_value": [16, 32, 64, 128]},
    "hidden_size":{"_type":"choice","_value":[[128,512,64,3],[1024,256]]},
    'lr':{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]},
    "momentum":{"_type":"uniform","_value":[0, 1]}
}
tuner_params = nni.get_next_parameter()