CUDA_VISIBLE_DEVICES=0 python main.py --config-file configs/coco/scsm_det_r101_base.yaml \
	OUTPUT_DIR "output/coco/base" \
        MODEL.BACKBONE.WITHSCSM True \
        # MODEL.BACKBONE.WITHECEA = False
        # MODEL.BACKBONE.FREEZE_ECEA = False
        # MODEL.BACKBONE.WITHMAMBA = False
        # MODEL.BACKBONE.WITHSCSM = False
        # MODEL.BACKBONE.WITHSFASA = False
        # MODEL.BACKBONE.WITHSFASENet = False
        # MODEL.BACKBONE.CNN = False
        # MODEL.BACKBONE.ATTACK = False
