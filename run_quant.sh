python test_quant.py \
    deit_small \
    /root/pubdatasets/ILSVRC2012/ \
    --quant --quant-method minmax \
    --lis \
    --use_checkpoint \
    --checkpoint_path /root/kyzhang/yjwang/PTQ4ViT/original/a_vit_small_patch16_224.pth

python test_quant.py \
    deit_tiny \
    /root/pubdatasets/ILSVRC2012/ \
    --quant --quant-method minmax \
    --lis \
    --use_checkpoint \
    --checkpoint_path /root/kyzhang/yjwang/PTQ4ViT/original/a_vit_tiny_patch16_224.pth