# Convert M-LSD tiny bản PyTorch (lhwcv/mlsd_pytorch, có mã nguồn) sang TFLite, với đầu vào / đầu ra
# GIỐNG HỆT bản NAVER chính thức mà app đang dùng, để app và các script đo dùng thẳng được:
#   vào: (1, 512, 512, 4) float32, RGB 0..255 + kênh toàn số 1 (NHWC)
#   ra : pts (1, 200, 2) int32 (y, x) | scores (1, 200) float32 | vmap (1, 256, 256, 4) float32
#
# Cần môi trường có torch + litert_torch (vd. "/Users/duc/Neural Network Quantization/3-deploy/.venv").
# Chạy từ thư mục gốc repo:  <python đó> tools/export_lhwcv.py
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import litert_torch
from ai_edge_litert.interpreter import Interpreter

sys.path.insert(0, "third_party/mlsd-pytorch")      # symlink tới clone lhwcv/mlsd_pytorch
from models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny

OUT_DIR = "data/models"


def load_net():
    net = MobileV2_MLSD_Tiny().eval()
    net.load_state_dict(torch.load("third_party/mlsd-pytorch/models/mlsd_tiny_512_fp32.pth", map_location="cpu"), strict=True)
    return net


class MlsdOfficialIO(nn.Module):
    # Gói model: chuẩn hoá đầu vào như lhwcv (x / 127.5 - 1, NCHW) và thêm phần cuối giống bản NAVER
    def __init__(self, net, topk=200):
        super().__init__()
        self.net, self.topk = net, topk

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) / 127.5 - 1.0
        out = self.net(x)                                     # (1, 9, 256, 256): kênh 0 = center, 1-4 = vmap
        center = torch.sigmoid(out[:, 0:1])
        hmax = F.max_pool2d(center, 3, stride=1, padding=1)   # giữ điểm cực đại trong ô 3x3
        center = center * (hmax == center).float()
        scores, idx = torch.topk(center.reshape(1, -1), self.topk)
        pts = torch.stack([idx // 256, idx % 256], dim=-1).to(torch.int32)
        vmap = out[:, 1:5].permute(0, 2, 3, 1)
        return pts, scores, vmap


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    model = MlsdOfficialIO(load_net()).eval()
    rng = np.random.default_rng(0)
    sample = np.concatenate([rng.integers(0, 256, (1, 512, 512, 3)), np.ones((1, 512, 512, 1))], -1).astype(np.float32)
    path = f"{OUT_DIR}/lhwcv_fp32.tflite"
    litert_torch.convert(model, (torch.from_numpy(sample),)).export(path)

    # Kiểm tra: TFLite khớp PyTorch, và thứ tự / kiểu đầu ra giống bản NAVER
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], sample)
    it.invoke()
    got = [it.get_tensor(d["index"]) for d in it.get_output_details()]
    with torch.no_grad():
        ref = [t.numpy() for t in model(torch.from_numpy(sample))]
    for d in it.get_output_details():
        print("OUT", d["index"], d["shape"], d["dtype"].__name__)
    print(f"pts trung {np.mean(got[0] == ref[0]) * 100:.1f}% | lech score max {np.abs(got[1] - ref[1]).max():.1e} "
          f"| lech vmap max {np.abs(got[2] - ref[2]).max():.1e}")
    print("Da luu", path, f"({os.path.getsize(path) / 1e6:.2f} MB)")
