"""Regenerate policy_deploy.onnx for rl_sar from a (already 57-dim) policy.onnx.

rl_sar feeds the ONNX a fixed-shape tensor and its inference runtime
(Ort::Session) fails on a symbolic batch dim, so we just freeze the first
(input) dimension to a static 1. The feature dim is already 57, the weights
are left untouched -- this is a faithful copy, NOT a truncation. (The legacy
truncate_to_41.py is only for the old 174-dim export and would wrongly slice a
hidden layer here, so don't use it for a 57-dim policy.)

Requires: pip install onnx
"""
import onnx

SRC = "policy.onnx"
DST = "policy_deploy.onnx"

m = onnx.load(SRC)
inp = m.graph.input[0]
d0 = inp.type.tensor_type.shape.dim[0]
d0.Clear()
d0.dim_value = 1
onnx.checker.check_model(m)
onnx.save(m, DST)

m2 = onnx.load(DST)
d = m2.graph.input[0].type.tensor_type.shape.dim
print(f"wrote {DST}: input shape = [{d[0].dim_value}, {d[1].dim_value}]")
