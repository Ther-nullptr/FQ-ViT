python test_quant.py \
    swin_tiny \
    /root/pubdatasets/ILSVRC2012/ \
    --quant --ptf --lis --quant-method minmax

python test_quant.py \
    swin_tiny \
    /root/pubdatasets/ILSVRC2012/ \
    --quant --ptf --quant-method minmax