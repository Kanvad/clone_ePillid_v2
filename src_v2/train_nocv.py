from __future__ import annotations

import os
import sys
import copy
import datetime
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms as T
from sklearn.preprocessing import LabelEncoder
import pickle
import logging

from arguments import nocv_parser, parse_bool_args, load_config
from multihead_trainer import train, torch_transform
from metric_test_eval import MetricEmbeddingEvaluator, LogitEvaluator
from sanitytest_eval import create_eval_dataloaders

logger = logging.getLogger(__name__)


class MockRun:
    """Mock Azure ML run for local execution."""

    def __init__(self):
        self._tags = {}

    def log(self, name: str, value: Any) -> None:
        logger.info(f"{name}: {value}")

    def log_image(self, name: str, plot: Any) -> None:
        logger.info(f"Logged image: {name}")

    def tag(self, key: str, value: str) -> None:
        self._tags[key] = value


def get_run() -> MockRun:
    """Get the Azure ML run context or a mock for local execution."""
    try:
        from azureml.core.run import Run
        return Run.get_context()
    except ImportError:
        return MockRun()


def run(args: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main training function."""
    if args.supress_warnings:
        warnings.simplefilter("ignore")

    args = parse_bool_args(args)

    def adjust_path(p: str) -> str:
        return os.path.join(args.data_root_dir, p)

    args.label_encoder = adjust_path(args.label_encoder)
    args.all_imgs_csv = adjust_path(args.all_imgs_csv)
    args.val_imgs_csv = adjust_path(args.val_imgs_csv)
    args.test_imgs_csv = adjust_path(args.test_imgs_csv)
    args.results_dir = adjust_path(args.results_dir)

    print(args)

    def build_logid_string(args: Any, add_timestamp: bool = True) -> str:
        param_str = "lr{}_dr{}_lrpatience{}_lrfactor{}_{}".format(
            args.init_lr,
            args.dropout,
            args.lr_patience,
            args.lr_factor,
            args.appearance_network,
        )

        if add_timestamp:
            param_str += "_" + datetime.datetime.now().strftime("%Y%m%d%H%M")

        return param_str

    param_str = build_logid_string(args)

    run = get_run()

    if not hasattr(args, "folds_csv_dir"):
        for k, v in vars(args).items():
            run.tag(k, str(v))

    save_path = os.path.join(args.results_dir, param_str)
    os.makedirs(save_path, exist_ok=True)
    print("save_path", save_path)

    logger.info(
        f"cuda.is_available={torch.cuda.is_available()}, n_gpu={torch.cuda.device_count()}"
    )

    if not os.path.exists(args.label_encoder):
        logger.warning(f"Fitting a new label encoder at {args.label_encoder}")

        all_imgs_df = pd.read_csv(args.all_imgs_csv)

        label_encoder = LabelEncoder()
        label_encoder.fit(all_imgs_df["label"])

        with open(args.label_encoder, "wb") as f:
            pickle.dump(label_encoder, f)
    else:
        logger.info(f"Loading label encoder: {args.label_encoder}")

        with open(args.label_encoder, "rb") as f:
            label_encoder = pickle.load(f)

    logger.info(f"label_encoder.classes_={label_encoder.classes_}")
    logger.info(f"The label encoder has {len(label_encoder.classes_)} classes.")

    all_images_df = pd.read_csv(args.all_imgs_csv)
    val_df = pd.read_csv(args.val_imgs_csv)
    test_df = pd.read_csv(args.test_imgs_csv)

    for df in [all_images_df, val_df, test_df]:
        df["image_path"] = df["image_path"].apply(
            lambda x: os.path.join(args.data_root_dir, args.img_dir, x)
        )

    val_test_image_paths = list(val_df["image_path"].values) + list(
        test_df["image_path"].values
    )
    train_df = all_images_df[~all_images_df["image_path"].isin(val_test_image_paths)]

    ref_only_df = train_df[train_df["is_ref"]]
    cons_train_df = train_df[train_df["is_ref"] == False]
    cons_val_df = val_df

    print(
        "all_images",
        len(all_images_df),
        "train",
        len(train_df),
        "val",
        len(val_df),
        "test",
        len(test_df),
    )
    run.log("all_images_size", len(all_images_df))
    run.log("train_size", len(train_df))
    run.log("val_size", len(val_df))
    run.log("test_size", len(test_df))

    print(
        "ref_only_df",
        len(ref_only_df),
        "cons_train_df",
        len(cons_train_df),
        "cons_val_df",
        len(cons_val_df),
    )

    import classif_utils

    classif_utils.ClassificationDataset.set_datadir(
        os.path.join(args.data_root_dir, args.img_dir)
    )

    def plot_pr_curve(plt: Any, dataset_name: str) -> None:
        run.log_image(
            name="{}_{}_{}".format(
                dataset_name,
                datetime.datetime.now().strftime("%H:%M:%S"),
                "PR-curve",
            ),
            plot=plt,
        )
        plt.close()

    def log_metrics(metrics_results: dict, dataset_name: str) -> list:
        from metrics import create_prec_inds_str

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        run_metrics = []

        for k, v in metrics_results.items():
            if "p_indices" in k and not ("sanity" in dataset_name):
                pind_str = create_prec_inds_str(v, label_encoder)

                run.log("{}_{}".format(dataset_name, k), pind_str)
                run_metrics.append(
                    [
                        os.path.split(args.val_imgs_csv)[1],
                        dataset_name,
                        k,
                        pind_str,
                    ]
                )

            elif isinstance(v, (int, float)):
                run.log("{}_{}".format(dataset_name, k), v)
                run_metrics.append(
                    [
                        os.path.split(args.val_imgs_csv)[1],
                        dataset_name,
                        k,
                        v,
                    ]
                )

        return run_metrics

    model, val_metrics = train(
        ref_only_df,
        cons_train_df,
        cons_val_df,
        label_encoder,
        torch_transform,
        "label",
        args.batch_size,
        len(label_encoder.classes_),
        args,
        args.max_epochs,
        results_dir=save_path,
        add_perspective=args.add_persp_aug,
    )

    print("completed train()")
    print("val_metrics", val_metrics)

    run_metrics_list = log_metrics(val_metrics, "val")
    predictions_dfs_list = []

    evaluator = MetricEmbeddingEvaluator(
        model,
        args.metric_simul_sidepairs_eval,
        sidepairs_agg_method=args.sidepairs_agg,
        metric_evaluator_type=args.metric_evaluator_type,
    )

    logit_evaluator = LogitEvaluator(
        model,
        args.metric_simul_sidepairs_eval,
        sidepairs_agg_method=args.sidepairs_agg,
    )

    def get_labelcol_eval(de_imgs_df: pd.DataFrame) -> pd.Series:
        labels_df = pd.DataFrame({"label": label_encoder.classes_})
        img_df = pd.merge(
            de_imgs_df,
            labels_df,
            left_on=["label_prod_code"],
            right_on=["label"],
            how="inner",
        )

        if len(img_df) > 1:
            labelcol = "label_prod_code"
        else:
            labelcol = "pilltype_id"
        print(f"Selecting {labelcol} for sanity test label")

        return de_imgs_df[labelcol]

    def test_model(
        de_imgs_df: pd.DataFrame,
        evaluator: Any,
        dataset_name: str,
        run_metrics_list: list,
        predictions_dfs_list: list,
        rotate_aug: Optional[int] = None,
    ) -> tuple[dict, pd.DataFrame]:
        if rotate_aug is not None:
            dataset_name += "_rotate_aug{}".format(rotate_aug)

        print("Evaluating", dataset_name)
        eval_dataloader, eval_dataset = create_eval_dataloaders(
            de_imgs_df,
            label_encoder,
            torch_transform,
            "label",
            24,
            rotate_aug=rotate_aug,
        )

        ref_dataloader, _ = create_eval_dataloaders(
            ref_only_df,
            label_encoder,
            torch_transform,
            "label",
            24,
            rotate_aug=rotate_aug,
        )
        dataloader = {"ref": ref_dataloader, "eval": eval_dataloader}

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(
            "Eval {}: {} images from {} total images".format(
                dataset_name, len(eval_dataset), len(de_imgs_df)
            )
        )

        metrics_results, predictions = evaluator.eval_model(
            device, dataloader, do_pr_metrics=True, add_single_side_eval=True
        )

        plot_pr_curve(metrics_results["PR-curve"], dataset_name)

        run_metrics_list += log_metrics(metrics_results, dataset_name)

        predictions["dataset"] = dataset_name
        predictions["val_imgs_csv"] = os.path.split(args.val_imgs_csv)[1]
        predictions_dfs_list.append(predictions)

        return metrics_results, predictions

    test_model(
        test_df, logit_evaluator, "holdout-logit", run_metrics_list, predictions_dfs_list
    )
    test_model(test_df, evaluator, "holdout", run_metrics_list, predictions_dfs_list)

    run_metrics_df = pd.DataFrame(
        run_metrics_list, columns=["val_imgs_csv", "dataset", "name", "value"]
    )
    all_predictions_df = pd.concat(predictions_dfs_list, ignore_index=True)

    for target_save_dir in [save_path, "outputs"]:
        print(f"saving predictions {target_save_dir}")
        os.makedirs(target_save_dir, exist_ok=True)
        all_predictions_df.to_csv(
            os.path.join(
                target_save_dir,
                "eval_predictions_{}".format(os.path.basename(args.val_imgs_csv)),
            )
        )

    torch.save(
        model.state_dict(),
        os.path.join(save_path, "{}.pth".format(os.path.basename(args.val_imgs_csv))),
    )

    return run_metrics_df, all_predictions_df


if __name__ == "__main__":
    import arguments

    args = arguments.nocv_parser().parse_args()

    if args.load_config:
        load_config(args)

    run_results, all_predictions_df = run(args)
