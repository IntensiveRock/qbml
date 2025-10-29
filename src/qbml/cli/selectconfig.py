# define the configs as dictionaries that can be exported.

setgen_cfg = {
    "seed": 12345,
    "symbol": "a",
    "runs": 1,
    "beta": 1,
    "save": False,
    "system": {
        "hbar": 1,
        "epsilon": "&eps 1",
        "delta": "&delta 1",
        "mineps": "&mineps -1",
        "hami": "[[*eps, *delta], [*delta, *mineps]]",
    },
    "sb": {"hami": [[[0, 1], [1, 0]], [[1, 0], [0, -1]]]},
    "specden": {
        "type": ["three_peak_lorentzian"],
        "params": {"insert": "parameter for chosen spectral density."},
    },
    "times": {"start": 0, "stop": 5, "dt": 0.01},
    "initial_condition": [0.5, 0.5, 0.5, 0.5],
}

nn_cfg = {
    "name" : "",
    "cfg_info" : {
        "device" : "cpu",
        "model"  : "",
        "loss_fn": "",
        "optimizer" : "",
    },
    "model_hyperparameters" : {
        "n_encoders" : 1,
        "batch_size" : 10,
        "learning_rate" : 1e-5,
        "weight_decay" : 0,
        "loss_nvp" : 1e-5,
        "epochs" : 100,
        "input_seq_len" : "",
        "target_seq_len" : "",
        "rectifier" : "",
        "block_size" : "",
    },
    "dataset" : {
        "split" : [0.9, 0.1],
    },
    "set_path" : "",
}
