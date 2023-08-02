import torch
import argparse
import yaml
import onnx
from pathlib import Path
from onnxsim import simplify
from framework.models import *
from framework.datasets import *
from framework.model import get_model

def export_onnx(model, inputs, file):
    torch.onnx.export(
        model,
        inputs,
        f"{cfg['TEST']['MODEL_PATH'].split('.')[0]}.onnx",
        input_names=['input'],
        output_names=['output'],
        opset_version=13
    )
    onnx_model = onnx.load(f"{file}.onnx")
    onnx.checker.check_model(onnx_model)

    onnx_model, check = simplify(onnx_model)
    onnx.save(onnx_model, f"{file}.onnx")
    assert check, "Simplified ONNX model could not be validated"
    print(f"ONNX model saved to {file}.onnx")


def export_coreml(model, inputs, file):
    try:
        import coremltools as ct
        ts_model = torch.jit.trace(model, inputs, strict=True)
        ct_model = ct.convert(
            ts_model,
            #todo: support
            inputs=[]#ct.ImageType('image', shape=inputs.shape, scale=1/255.0, bias=[0, 0, 0])]
        )
        ct_model.save(f"{file}.mlmodel")
        print(f"CoreML model saved to {file}.mlmodel")
    except:
        print("Please install coremltools to export to CoreML.\n`pip install coremltools`")
    

def main(cfg):
    model = get_mode(cfg['MODEL'])
    model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu'))
    model.eval()
    #todo: support
    inputs = torch.randn(1, 3, *cfg['TEST']['INPUT'])
    file = cfg['TEST']['MODEL_PATH'].split('.')[0]

    export_onnx(model, inputs, file)
    export_coreml(model, inputs, file)
    print(f"Finished converting.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    
    main(cfg)