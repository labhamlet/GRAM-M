from hear_api.runtime import RuntimeGRAMM
import torch


class Config:
     def __init__(self, dictionary):
         for k, v in dictionary.items():
             setattr(self, k, v)

MODEL_PATH = "/gpfs/work4/0/prjs1338/saved_models/Data=AudioSet/WithNoise=True/Model=SpatialMamba/ModelSize=base/NrGpus=1/LR=0.0002/BatchSize=32/NrSamples=16/Patching=frame/MaskPatch=200/InputL=200/Cluster=False/step=91000.ckpt"

config = Config({"model": Config({"mlp_ratio": 4.0,
                    "d_state": 24 ,
                    "d_conv": 4,
                    "expand": 3, 
                    "rms_norm": True ,
                    "residual_in_fp32": True , 
                    "fused_add_norm": True, 
                    "norm_epsilon": 1e-6}),
            "trainer": Config({"precision": "bf16-true"})})

def load_model(*args, **kwargs):
    if len(args) != 0:
        model_path = args[0]
    else:
        model_path = MODEL_PATH
    strategy = kwargs.get("strategy", "raw")
    use_mwmae_decoder = str(kwargs.get("use_mwmae_decoder", False)) == "true"
    in_channels = kwargs.get("in_channels", 2)
    layer = kwargs.get("layer", None)
    model = RuntimeGRAMM(model_size="base",
                                   config=config,
                                   in_channels=in_channels,
                                   weights=torch.load(model_path),
                                   fshape=16,
                                   tshape=8,
                                   fstride=16,
                                   tstride=8,
                                   input_tdim=200,
                                   use_mwmae_decoder = use_mwmae_decoder,
                                   decoder_window_sizes = [2,5,10,25,50,100,0,0],
                                   strategy= strategy,
                                   layer = layer)
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
