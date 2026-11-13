"""从项目 pretrained/ 读取预训练权重，训练时不联网或访问用户权重缓存。"""
from pipeline_common import project_path


def load_cnn_pretrained(model, model_name):
    """加载原始 ImageNet 分类头，调用者随后将分类头替换为八分类。"""
    import torch

    filenames = {
        "efficientnet": ("efficientnet_b0_rwightman-3dd342df.pth",
                         "efficientnet_b0_rwightman-7f5810bc.pth"),
        "resnet101": ("resnet101-63fe2227.pth",),
    }[model_name]
    paths = [project_path("pretrained/" + name) for name in filenames]
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "缺少本地预训练权重，请将下载文件放入项目目录: "
            + " 或 ".join("pretrained/" + name for name in filenames)
            + "。不会自动下载到 C 盘。")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # YuHou 的 PyTorch 1.12 不支持 weights_only。
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=True)


def local_vit_directory():
    directory = project_path("pretrained/vit-base-patch16-224-in21k")
    required = ("config.json", "preprocessor_config.json")
    if (not all((directory / name).is_file() for name in required)
            or not any((directory / name).is_file()
                       for name in ("pytorch_model.bin", "model.safetensors"))):
        raise FileNotFoundError(
            "请将 ViT 的 config.json、preprocessor_config.json 和 "
            "pytorch_model.bin（或 model.safetensors）放入项目 "
            "pretrained/vit-base-patch16-224-in21k/，然后重新运行。")
    return directory
