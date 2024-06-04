
import onnx
import onnxruntime
import onnxruntime as ort
import numpy as np

def main():
    model_path = 'end2end.onnx'
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)


    # x, y = test_data[0][0], test_data[0][1]
    ort_sess = ort.InferenceSession(model_path)

if __name__ == "__main__":
    main()