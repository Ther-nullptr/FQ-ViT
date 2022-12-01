python test_quant.py \
    deit_small \
    /root/pubdatasets/ILSVRC2012/ \
    --quant --ptf --quant-method minmax --checkpoint \
    --checkpoint_path /root/kyzhang/yjwang/PTQ4ViT/original/a_vit_small_patch16_224.pth \
    --name avit
