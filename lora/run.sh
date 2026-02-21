# Training with 4-bit quantization + deblur module
# python lora/train.py --epochs 20 --batch_size 4 --lr 2e-4

# Hoặc dùng ESRGAN-lite deblur mạnh hơn
# python lora/train.py --deblur_type esrgan_lite --epochs 20

# Không dùng deblur (so sánh)
# python lora/train.py --no_deblur --epochs 20

# python lora/train.py --mode inference --output_dir results/paligemma_lora

# Tối ưu cho 16GB VRAM
# python lora/train.py --epochs 20 --batch_size 8 --gradient_accumulation 2 --lr 2e-4 --lora_r 8 --lora_alpha 16

# Tối ưu cho 16GB VRAM + Flash Attention 2
# ...existing code...

# Fixed training: lower LR, more epochs, more data, debug output
python lora/train.py --epochs 20 --batch_size 8 --gradient_accumulation 2 --lr 1e-4 --lora_r 8 --lora_alpha 16 --data_limit 5000
