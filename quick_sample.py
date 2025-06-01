from models.dit import MFDiT
from meanflow import MeanFlow
import torch
from torchvision.utils import make_grid, save_image
import os

def load_model_and_infer(
    ckpt_path='checkpoints/step_20000.pt',
    output_path='inference_output.png',
    device='cuda',
    num_classes=10
):
    # 1. 初始化模型和 MeanFlow
    model = MFDiT(
        input_size=32,
        patch_size=2,
        in_channels=3,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
    ).to(device)

    meanflow = MeanFlow(
        channels=3,
        image_size=32,
        num_classes=num_classes,
        flow_ratio=0.50,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        cfg_scale=2.0,
        cfg_uncond='u',
    )

    # 2. 加载 checkpoint
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. 推理生成每类图像
    with torch.no_grad():
        z = meanflow.sample_each_class(model,10)  # 每类生成1个图像
        log_img = make_grid(z, nrow=10)  # 10类 → 1 行 10 张图

    # 4. 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(log_img, output_path)
    print(f"[✓] Saved inference result to {output_path}")
    # sample with cfg
    with torch.no_grad():
        z = meanflow.sample_each_class_with_cfg(model,10)  # 每类生成1个图像
        log_img = make_grid(z, nrow=10)  # 10类 → 1 行 10 张图
    output_path = output_path.replace('.png', '_cfg.png')
    # 4. 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(log_img, output_path)
    print(f"[✓] Saved inference result to {output_path}")


if __name__ == '__main__':
    # load_model_and_infer(
    #     ckpt_path="checkpoints/fd_step_200000.0.pt",
    #     output_path="./inference/fd_step_200000_inference_output.png",
    # )
    load_model_and_infer(
        ckpt_path="checkpoints/jvp_step_200000.0.pt",
        output_path="./inference/jvp_step_200000_inference_output.png",
    )
