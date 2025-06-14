from models.dit import MFDiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import sys 
training_type = sys.argv[1] if len(sys.argv) > 1 else "fd"  # Default to 'fd' if no argument is provided
# training_type="jvp"
# training_type="fd" 
if training_type == "jvp": 
    print("Using JVP Training meanflow.py for training")
    from meanflow_jvp import MeanFlow
else:
    print("Using Finite Difference Training meanflow.py for training")
    from meanflow import MeanFlow
from accelerate import Accelerator
import time
import os


if __name__ == '__main__':
    n_steps = 200000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 48
    os.makedirs(f'{training_type}_images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16')

    dataset = torchvision.datasets.CIFAR10(
        root="cifar",
        train=True,
        download=True,
        transform=T.Compose([T.ToTensor(), T.RandomHorizontalFlip()]),
    )
    dataset_len = len(dataset)
    # dataset = torchvision.datasets.MNIST(
    #     root="mnist",
    #     train=True,
    #     download=True,
    #     transform=T.Compose([T.Resize((32, 32)), T.ToTensor(),]),
    # )

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8,pin_memory=True
    )
    train_dataloader = cycle(train_dataloader)

    model = MFDiT(
        input_size=32,
        patch_size=2,
        in_channels=3,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,
    ).to(accelerator.device)
    # model = MFDiT(
    #     input_size=32,
    #     patch_size=2,
    #     in_channels=1,
    #     dim=384,
    #     depth=12,
    #     num_heads=6,
    #     num_classes=10,
    # ).to(accelerator.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    meanflow = MeanFlow(channels=3,
                        image_size=32,
                        num_classes=10,
                        flow_ratio=0.50,
                        time_dist=['lognorm', -0.4, 1.0],
                        cfg_ratio=0.10,
                        cfg_scale=2.0,
                        # experimental
                        cfg_uncond='u')

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    global_step = 0.0
    losses = 0.0
    mse_losses = 0.0

    log_step = 500
    sample_step = 500

    pbar = range(n_steps)
    if accelerator.is_main_process:
        pbar = tqdm(pbar, dynamic_ncols=True)
    for step in pbar:
        epoch = step // (dataset_len // batch_size)
        model.train()
        data = next(train_dataloader)
        x = data[0].to(accelerator.device)
        c = data[1].to(accelerator.device)

        loss, mse_val = meanflow.loss(model, x, c)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        losses += loss.item()
        mse_losses += mse_val.item()
        pbar.set_description(f"{training_type} Training (epoch: {epoch}) (loss: {loss:.4f})")
        if accelerator.is_main_process:
            if global_step % log_step == 0:
                current_time = time.asctime(time.localtime(time.time()))
                batch_info = f'Global Step: {global_step}'
                loss_info = f'Loss: {losses / log_step:.6f}    MSE_Loss: {mse_losses / log_step:.6f}'

                # Extract the learning rate from the optimizer
                lr = optimizer.param_groups[0]['lr']
                lr_info = f'Learning Rate: {lr:.6f}'

                log_message = f'{current_time}\n{batch_info}    {loss_info}    {lr_info}\n'

                with open(f'{training_type}_log.txt', mode='a') as n:
                    n.write(log_message)

                losses = 0.0
                mse_losses = 0.0

        if global_step % sample_step == 0:
            if accelerator.is_main_process:
                model_module = model.module if hasattr(model, 'module') else model
                z = meanflow.sample_each_class(model_module, 10)
                log_img = make_grid(z, nrow=10)
                img_save_path = f"{training_type}_images/step_{global_step}.png"
                save_image(log_img, img_save_path)
            accelerator.wait_for_everyone()
            model.train()
            
        if accelerator.is_main_process and global_step % 1000 == 0:
            ckpt_path = f"checkpoints/{training_type}_step_{global_step}.pt"
            accelerator.save(model_module.state_dict(), ckpt_path)