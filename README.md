# motion-blur-SR

## Dependenices
-  einops ==0.6.0
-   ninja == 1.11.1
-   pytorch == 2.0.1

## Pretrained Model
Please go to [https://drive.google.com/file/d/1pKw_8xxeSqE4lzOPLSMlHCVGHvCLssqg/view?usp=sharing](https://drive.google.com/file/d/1pKw_8xxeSqE4lzOPLSMlHCVGHvCLssqg/view?usp=sharing) to download the pretrained models.
## Train & Evaluate
1.  Modify config files.
2.  Run training / evaluation code. The code is for training on 1 GPU.

### Train
	python3 blur_sr_up.py --exp blur_sr --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 16 --num_epoch 75 --ngf 64 --nz 10 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --num_process_per_node 1 --ch_mult 1 2 3 4 --save_content --rec_loss --opt ./options/setting1/train/train_setting1_x4.yml
### Evaluate
	python3 test_ddgan_blur_sr.py --exp blur_sr_gopro --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 3 4 --epoch_id 18 --opt ./options/setting1/test/test_setting1_x4.yml

### Some Results

[![motion-blur-SR](https://github.com/tonia86/motion-blur-SR/result.png)](https://github.com/tonia86/motion-blur-SR/result.png)

## Acknowledgments

This code is mainly built on [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement), [DDGAN](https://github.com/NVlabs/denoising-diffusion-gan.git).
