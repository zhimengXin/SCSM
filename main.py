import os
from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from scsm.config import get_cfg, set_global_cfg
from scsm.evaluation import DatasetEvaluators, verify_results
from scsm.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.data import DatasetMapper
# from detectron2.config import get_cfg, set_global_cfg
# from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup



class Trainer(DefaultTrainer):

    @classmethod   
            
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            from defrcn.evaluation import COCOEvaluator
            evaluator_list.append(COCOEvaluator(dataset_name, True, output_folder))
        if evaluator_type == "pascal_voc":
            from defrcn.evaluation import PascalVOCDetectionEvaluator
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    # if args.data_aug != "None":
    #     cfg.MODEL.BACKBONE.DATA_AUG_TYPE = args.data_aug
    # if args.feature_aug != "None":
    #     cfg.MODEL.BACKBONE.FEATURE_AUG_TYPE = args.feature_aug
    # if args.output_dir != "":
    #     cfg.OUTPUT_DIR = args.output_dir
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):

    # app = Flask(__name__)
    # app.run(port=2004)

    cfg = setup(args)
    # args.resume = True    

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

   
        
    trainer = Trainer(cfg)


    trainer.resume_or_load(resume=args.resume)


    data_loader = trainer.data_loader
    print(data_loader)
    # print("类名：", data_loader.dataset.classes)
    # print("类的个数：", len(data_loader.dataset.classes))

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
