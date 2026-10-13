# Mốc 5, bước 2-3: tạo các bản INT8 (static) của M-LSD tiny bản lhwcv, mỗi bản thêm 1 cách sửa,
# để thấy từng cách sửa giúp bao nhiêu. Đầu vào / đầu ra giống bản NAVER (xem tools/export_lhwcv.py).
#   V1 INT8 mặc định                 (head 9 kênh chung 1 tensor -> chung 1 scale)
#   V2 + tách head                   (center và vmap là 2 conv / 2 tensor riêng -> 2 scale riêng)
#   V3 + cắt ngoại lai               (center >= -8, vmap trong ±128: bỏ giá trị rác ở chỗ không có đoạn thẳng)
#   V4 + giữ 3 layer nhạy ở float    (3 conv đầu tiên, theo thí nghiệm 7-mlsd-lowbit)
#   V5 + giữ phần head ở float
# Calibration: data/calib.txt (360 khung SmartDoc, tools/make_doc_split.py).
#
# Cần môi trường có torch + litert_torch + ai_edge_quantizer (vd. "/Users/duc/Neural Network Quantization/3-deploy/.venv").
# Chạy từ thư mục gốc repo:  <python đó> tools/quantize_lhwcv.py
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import litert_torch
from ai_edge_quantizer import algorithm_manager, quantizer, recipe
from ai_edge_litert.interpreter import Interpreter

sys.path.insert(0, "tools")
from export_lhwcv import MlsdOfficialIO, load_net, OUT_DIR

C_LO, V_R = -8.0, 128.0      # khoảng giữ lại cho center (logit) và vmap, xem tools/sim_quant_head.py


def decode(center_logit, vmap_nchw, topk=200):
    # Phần cuối giống bản NAVER: sigmoid -> giữ cực đại 3x3 -> top-k -> pts (y, x), scores, vmap NHWC
    center = torch.sigmoid(center_logit)
    hmax = F.max_pool2d(center, 3, stride=1, padding=1)
    center = center * (hmax == center).float()
    scores, idx = torch.topk(center.reshape(1, -1), topk)
    pts = torch.stack([idx // 256, idx % 256], dim=-1).to(torch.int32)
    return pts, scores, vmap_nchw.permute(0, 2, 3, 1)


class MlsdSplitIO(nn.Module):
    # Tách conv cuối (block16.conv3, 16 kênh ra) thành 2 conv: center (kênh 7) và vmap (kênh 8-11).
    # Kết quả FP32 không đổi, nhưng khi quantize mỗi phần có tensor (và scale) riêng.
    def __init__(self, net, clip=False):
        super().__init__()
        self.net, self.clip = net, clip
        src = net.block16.conv3

        def sub_conv(rows):
            conv = nn.Conv2d(src.in_channels, len(rows), 1)
            conv.weight.data = src.weight.data[rows].clone()
            conv.bias.data = src.bias.data[rows].clone()
            return conv
        self.center_conv = sub_conv([7])
        self.vmap_conv = sub_conv([8, 9, 10, 11])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) / 127.5 - 1.0
        m = self.net
        c2, c3, c4 = m.backbone(x)
        y = m.block15(m.block14(c2, m.block13(m.block12(c3, c4))))
        y = m.block16.conv2(m.block16.conv1(y))
        c = self.center_conv(y)
        v = self.vmap_conv(y)
        if self.clip:
            # Cắt ngoại lai: những giá trị này chỉ có ở chỗ không có đoạn thẳng, không ảnh hưởng kết quả
            c = torch.clamp(c, min=C_LO)
            v = torch.clamp(v, -V_R, V_R)
        up = lambda t: F.interpolate(t, scale_factor=2.0, mode="bilinear", align_corners=True)
        return decode(up(c), up(v))


def calib_data(sig_input, n=None):
    paths = [l.strip() for l in open("data/calib.txt")][:n]
    for p in paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        x = np.concatenate([cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA),
                            np.ones([512, 512, 1])], axis=-1)[None].astype(np.float32)
        yield {sig_input: x}


def export_fp32(model, name):
    sample = torch.from_numpy(np.concatenate([np.full((1, 512, 512, 3), 128.0), np.ones((1, 512, 512, 1))], -1).astype(np.float32))
    path = f"{OUT_DIR}/{name}.tflite"
    litert_torch.convert(model.eval(), (sample,)).export(path)
    return path


def conv_ops(path):
    # Tên tensor output của các op conv, theo thứ tự trong đồ thị
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    td = it.get_tensor_details()
    return [td[o["outputs"][0]]["name"] for o in it._get_ops_details() if o["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D")]


def no_q(name):
    import re
    # tên trong ai_edge_quantizer = tên tensor output của op + ";" (giống 7-mlsd-lowbit/exp2)
    return {"regex": "^" + re.escape(name) + ";$", "operation": "*",
            "algorithm_key": algorithm_manager.AlgorithmName.NO_QUANTIZE, "op_config": {}}


def quantize(fp32_path, out_name, keep_float=()):
    rcp = recipe.static_wi8_ai8() + [no_q(n) for n in keep_float]
    qt = quantizer.Quantizer(fp32_path, rcp)
    sig_name, sig = list(Interpreter(model_path=fp32_path).get_signature_list().items())[0]
    cal = qt.calibrate({sig_name: list(calib_data(sig["inputs"][0]))})
    path = f"{OUT_DIR}/{out_name}.tflite"
    qt.quantize(cal).export_model(path, overwrite=True)
    return path


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    base = export_fp32(MlsdOfficialIO(load_net()), "lhwcv_fp32")
    split = export_fp32(MlsdSplitIO(load_net()), "lhwcv_split_fp32")
    clip = export_fp32(MlsdSplitIO(load_net(), clip=True), "lhwcv_splitclip_fp32")

    convs = conv_ops(clip)
    sensitive = convs[:3]                         # conv đầu backbone + depthwise / pointwise của InvertedResidual_1
    head = convs[-2:]                             # center_conv, vmap_conv
    print("3 layer nhay:", [n.split(";")[0][-60:] for n in sensitive])
    print("head:", [n.split(";")[0][-60:] for n in head])

    out = [quantize(base, "lhwcv_v1_int8"),
           quantize(split, "lhwcv_v2_split_int8"),
           quantize(clip, "lhwcv_v3_splitclip_int8"),
           quantize(clip, "lhwcv_v4_splitclip_keep3_int8", keep_float=sensitive),
           quantize(clip, "lhwcv_v5_splitclip_keep3head_int8", keep_float=sensitive + head)]
    for p in out:
        it = Interpreter(model_path=p)
        io = [(d["dtype"].__name__, list(d["shape"])) for d in it.get_input_details() + it.get_output_details()]
        print(f"{p}: {os.path.getsize(p) / 1e6:.2f} MB, vao/ra {io}")
