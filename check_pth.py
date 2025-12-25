import torch

path = "logs/transoss_fuse/transformer_100.pth"

checkpoint = torch.load(path, map_location="cpu")

if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    print("检测到 'state_dict' 键，已提取子字典。")
elif "model" in checkpoint:
    state_dict = checkpoint["model"]
    print("检测到 'model' 键，已提取子字典。")
else:
    state_dict = checkpoint
    print("未检测到嵌套结构，直接作为 state_dict 处理。")

print(f"\n=== 总共有 {len(state_dict)} 个 Key ===")
for key in state_dict.keys():
    print(key)
