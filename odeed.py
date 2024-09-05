# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Minimal standalone example to reproduce the main results from the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib

#----------------------------------------------------------------------------
def odeed_sampler(
    net, x, class_labels=None, 
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=x.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    t_steps = torch.cat([t_steps[:1:-1], t_steps])# t_N, ..., t_1(拡散過程), t_0, t_1, ..., t_N(復元過程)

    # Main sampling loop.
    #x_next = latents.to(torch.float64) * t_steps[0]
    x_next = x # ノイズなしの画像を初期値とする
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # PF-ODEの決定論的な実装
        # Euler step.
        denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def reconstruct_image_with_odeed(
    network_pkl, dataset_path, dest_path, seed=0, device=torch.device('cuda'),
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
):
    torch.manual_seed(seed)

    # Load network.(ここ変更必要、modelのloadをfile pathからできるようにする)
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    # Load dataset.(ここ変更必要、trainと同じノリでloadして出力できるようにする)
    print(f'Loading dataset from "{dataset_path}"...')
    dataset = torch.load(dataset_path)
    latents = dataset['latents'].to(device)
    class_labels = dataset.get('class_labels', None)
    idx = np.random.randint(len(latents))

    # Generate image.(ここ変更必要、生成した画像を適切なフォルダ階層で保存できるようにする)
    print(f'Generating image...')
    image = odeed_sampler(net, latents[idx:idx+1], class_labels[idx:idx+1], num_steps, sigma_min, sigma_max, rho)
    image = (image * 127.5 + 128).clip(0, 255).to(torch.uint8)
    image = image.permute(0, 2, 3, 1).cpu().numpy()
    PIL.Image.fromarray(image[0], 'RGB').save(dest_path)
    print('Done.')
    

def main():
    model_root = 'models'
    reconstruct_image_with_odeed(f'{model_root}/edm-spacenet8-256x256-uncond-vp????.pkl',   'output',  num_steps=18) # FID = 1.79, NFE = 35

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
