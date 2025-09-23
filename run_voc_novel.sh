gpu_id=$3
echo "gpu_id: ${gpu_id}"
for shot in 1 2 3 5 10
do 
    config_file="configs/voc/scsm_gfsod_r101_novel1_${shot}shot_seed1.yaml"
    CUDA_VISIBLE_DEVICES=0 python main.py --config-file ${config_file} \
        --opts \
        OUTPUT_DIR "output/${shot}shotgfsod" \
        MODEL.WEIGHTS "model_reset_surgery.pth" \
        MODEL.BACKBONE.WITHSCSM True \
        # MODEL.BACKBONE.WITHECEA = False
        # MODEL.BACKBONE.FREEZE_ECEA = False
        # MODEL.BACKBONE.WITHMAMBA = False
        # MODEL.BACKBONE.WITHSCSM = False
        # MODEL.BACKBONE.WITHECEASA = False
        # MODEL.BACKBONE.WITHECEASENet = False
        # MODEL.BACKBONE.CNN = False
done
