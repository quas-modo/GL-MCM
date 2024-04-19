EXP_NAME='exp1'
ID='ImageNet'
DATA_ROOT='/home/nfs03/zengtc'


CKPT=ViT-B/16

CUDA_VISIBLE_DEVICES=0,1,2,3 python tip_ood_detection.py --in_dataset ${ID} --name ${EXP_NAME} --CLIP_ckpt ${CKPT} --score MCM --root-dir ${DATA_ROOT}
