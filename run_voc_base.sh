CUDA_VISIBLE_DEVICES=0 python main.py --config-file configs/voc/scsm_det_r101_base1.yaml \
	--opts \
	OUTPUT_DIR "output/base1" \
        MODEL.BACKBONE.WITHSCSM True \
        # MODEL.BACKBONE.WITHECEA = False
        # MODEL.BACKBONE.FREEZE_ECEA = False
        # MODEL.BACKBONE.WITHMAMBA = False
        # MODEL.BACKBONE.WITHSCSM = False
        # MODEL.BACKBONE.WITHECEASA = False
        # MODEL.BACKBONE.WITHECEASENet = False
        # MODEL.BACKBONE.CNN = False
