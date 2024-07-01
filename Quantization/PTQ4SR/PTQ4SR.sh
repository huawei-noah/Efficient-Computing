# EDSR_Lx4
python basicsr/ptq4sr.py -opt options/ptq/EDSR/test_EDSR_Lx4_ptq_w8a8.yml
python basicsr/ptq4sr.py -opt options/ptq/EDSR/test_EDSR_Lx4_ptq_w6a6.yml
python basicsr/ptq4sr.py -opt options/ptq/EDSR/test_EDSR_Lx4_ptq_w4a4.yml

# EDSR_Lx2
python basicsr/ptq4sr.py -opt options/ptq/EDSR/test_EDSR_Lx2_ptq_w8a8.yml
python basicsr/ptq4sr.py -opt options/ptq/EDSR/test_EDSR_Lx2_ptq_w6a6.yml
python basicsr/ptq4sr.py -opt options/ptq/EDSR/test_EDSR_Lx2_ptq_w4a4.yml

# SRResNet_x4
python basicsr/ptq4sr.py -opt options/ptq/SRResNet/test_MSRResNet_x4_ptq_w8a8.yml
python basicsr/ptq4sr.py -opt options/ptq/SRResNet/test_MSRResNet_x4_ptq_w6a6.yml
python basicsr/ptq4sr.py -opt options/ptq/SRResNet/test_MSRResNet_x4_ptq_w4a4.yml

# SRResNet_x4
python basicsr/ptq4sr.py -opt options/ptq/SRResNet/test_MSRResNet_x2_ptq_w8a8.yml
python basicsr/ptq4sr.py -opt options/ptq/SRResNet/test_MSRResNet_x2_ptq_w6a6.yml
python basicsr/ptq4sr.py -opt options/ptq/SRResNet/test_MSRResNet_x2_ptq_w4a4.yml