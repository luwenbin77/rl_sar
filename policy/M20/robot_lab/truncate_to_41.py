import sys
"""Truncate a policy ONNX to its first N effective input dims (temporary helper).

The exported M20 policy takes a 174-dim input, but structure analysis of
actor.0.weight shows only columns 0..40 carry learned signal (column norms
~2-9); columns 41..173 sit at the random-init baseline (~1.30), i.e. those
inputs were zero/unused during training.  Feeding anything there adds
untrained random-weight contributions, so we drop them: keep columns 0..40
and shrink the graph input 174 -> 41 (== padding with zeros, exactly).
"""
import numpy as np

SRC = "policy.onnx"
DST = "policy_deploy.onnx"
KEEP = int(sys.argv[1]) if len(sys.argv) > 1 else 57


def enc_varint(n):
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def dec_varint(b, i):
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        if not x & 0x80:
            return r, i
        s += 7


def parse(buf, descend):
    out, i, n = [], 0, len(buf)
    while i < n:
        tag, i = dec_varint(buf, i)
        f, w = tag >> 3, tag & 7
        if w == 0:
            v, i = dec_varint(buf, i)
            out.append([f, w, v, None])
        elif w == 2:
            l, i = dec_varint(buf, i)
            payload = buf[i:i + l]
            i += l
            kids = None
            if f in descend and descend[f] is not None:
                try:
                    kids = parse(payload, descend[f])
                except Exception:
                    kids = None
            out.append([f, w, payload, kids])
        elif w == 5:
            out.append([f, w, buf[i:i + 4], None]); i += 4
        elif w == 1:
            out.append([f, w, buf[i:i + 8], None]); i += 8
        else:
            raise ValueError("wire %d" % w)
    return out


def ser(items):
    out = b""
    for f, w, v, kids in items:
        tag = enc_varint((f << 3) | w)
        if kids is not None:
            payload = ser(kids)
        elif w == 2:
            payload = v
        elif w == 0:
            payload = enc_varint(v)
        else:
            payload = v
        if w == 2:
            out += tag + enc_varint(len(payload)) + payload
        else:
            out += tag + payload
    return out


# input ValueInfoProto: 1=type(TypeProto) -> 1=tensor_type(Tensor) -> 2=shape -> 1=dim -> 1=dim_value
DIMENSION = {}
SHAPE = {1: DIMENSION}
TENSOR = {2: SHAPE}
TYPEPROT = {1: TENSOR}
VALUEINFO = {2: TYPEPROT}  # name=1, type=2
SAMPLE = {1:None,2:None,3:None}  # noqa
# TensorProto: descend so we can rebuild; NOTE field 9 is raw bytes -> keep opaque
TENSORPROTO = {}
GRAPH = {11: VALUEINFO, 5: TENSORPROTO}
MODEL = {7: GRAPH}

data = open(SRC, "rb").read()
root = parse(data, MODEL)

changed_input = False
sliced_weight = False
for f, w, v, kids in root:
    if f != 7 or kids is None:
        continue
    graph = kids
    for item in graph:
        f1, w1, v1, k1 = item
        # graph input dims
        if f1 == 11 and k1 is not None:
            for t in k1:
                if t[0] == 2 and t[3] is not None:          # type
                    for tt in t[3]:
                        if tt[0] == 1 and tt[3] is not None:  # tensor_type
                            for ten in tt[3]:
                                if ten[0] == 2 and ten[3] is not None:  # shape
                                    dims = ten[3]
                                    if len(dims) == 2:
                                        last = dims[1]
                                        assert last[0] == 1 and last[3] is not None
                                        old = last[3][0][2]
                                        last[3][0][2] = KEEP
                                        print(f"graph input dim {old} -> {KEEP}")
                                        # batch dim: dynamic (-1 / dim_param) -> static 1,
                                        # rl_sar feeds the model shape straight into
                                        # Ort::Value::CreateTensor (inference_runtime.cpp:193)
                                        # and would fail on a negative dim
                                        first = dims[0]
                                        kids = parse(first[2], {})
                                        first[3] = [[1, 0, 1, None]]   # dim_value = 1
                                        first[2] = ser(first[3])
                                        changed_input = True
        # initializers
        if f1 == 5 and k1 is not None:
            name = None
            for tp in k1:
                if tp[0] == 8:
                    name = tp[2].decode()
            if name and name.endswith("weight"):
                # rewrite dims + raw_data for the first Linear only
                for tp in k1:
                    if tp[0] == 1 and tp[1] == 0:  # dims are unpacked varints here
                        pass
                dims = [tp[2] for tp in k1 if tp[0] == 1 and tp[1] == 0]
                raw = [tp for tp in k1 if tp[0] == 9]
                if len(dims) == 2 and raw and not sliced_weight:
                    r, c = dims
                    arr = np.frombuffer(raw[0][2], "<f4").reshape(r, c)
                    if c > KEEP:
                        new = arr[:, :KEEP].astype("<f4").tobytes()
                        raw[0][2] = new
                        # fix the dims varints
                        dim_items = [tp for tp in k1 if tp[0] == 1 and tp[1] == 0]
                        dim_items[0][2] = r
                        dim_items[1][2] = KEEP
                        print(f"{name}: [{r}, {c}] -> [{r}, {KEEP}]")
                        sliced_weight = True

assert changed_input and sliced_weight, "patch targets not found"
out = ser(root)
open(DST, "wb").write(out)
print("wrote", DST, len(out), "bytes")

# self-check: reparse
root2 = parse(out, MODEL)
for f, w, v, kids in root2:
    if f != 7:
        continue
    for item in kids:
        if item[0] == 5 and item[3] is not None:
            nm = [t[2].decode() for t in item[3] if t[0] == 8]
            raw = [t[2] for t in item[3] if t[0] == 9]
            if raw:
                print("  verify initializer:", nm, "bytes", len(raw[0]), "-> floats", len(raw[0]) / 4)
