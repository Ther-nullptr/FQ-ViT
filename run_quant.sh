python  test_quant.py \
    deit_tiny \
    /data/ImageNet \
    --quant --quant-method minmax \
    --ptf \
    --lis \
    --use_checkpoint \
    --checkpoint_path /home/yujin/projects/PTQ4ViT/original/a_vit_tiny_patch16_224.pth

# python test_quant.py \
#     deit_tiny \
#     /root/pubdatasets/ILSVRC2012/ \
#     --quant --quant-method minmax \
#     --lis \
#     --use_checkpoint \
#     --checkpoint_path /root/kyzhang/yjwang/PTQ4ViT/original/a_vit_tiny_patch16_224.pth