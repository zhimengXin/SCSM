gpu_id=$3
echo "gpu_id: ${gpu_id}"
for shot in 10 30 
do
    config_file="configs/coco_swin/scsm_fsod_r101_novel_${shot}shot_seedx.yaml"
    CUDA_VISIBLE_DEVICES=0 python main.py --config-file ${config_file} \
        --num-gpus  0 \
        --opts \
        OUTPUT_DIR "coco/fsod/${shot}shot" \
        MODEL.WEIGHTS "coco/model_reset_remove.pth" \
        MODEL.BACKBONE.WITHSCSM True \
        # MODEL.BACKBONE.WITHECEA = False
        # MODEL.BACKBONE.FREEZE_ECEA = False
        # MODEL.BACKBONE.WITHMAMBA = False
        # MODEL.BACKBONE.WITHSCSM = False
        # MODEL.BACKBONE.WITHSFASA = False
        # MODEL.BACKBONE.WITHSFASENet = False
        # MODEL.BACKBONE.CNN = False
        # MODEL.BACKBONE.ATTACK = False
        
done