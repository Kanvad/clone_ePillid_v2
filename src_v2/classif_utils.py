from __future__ import annotations

import os
import inspect
import copy
from glob import glob
from typing import Optional, Any

import pandas as pd
import numpy as np


class ClassificationDataset:
    """Helper class for accessing pill classification related resources/images.

    Note: Using Singleton class to avoid large refactoring. There are several
    files that are directly calling methods in this module.
    """

    _singleton_instance: Optional["ClassificationDataset"] = None

    @staticmethod
    def get_instance() -> "ClassificationDataset":
        if ClassificationDataset._singleton_instance is None:
            ClassificationDataset._singleton_instance = ClassificationDataset()
        return ClassificationDataset._singleton_instance

    @staticmethod
    def set_datadir(data_dir: str) -> None:
        ClassificationDataset._singleton_instance = ClassificationDataset(data_dir)

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            curr_dir = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
            data_rel = os.path.normpath("../data")

            data_dir = os.path.join(curr_dir, data_rel)

        print("Configuring data_dir", data_dir)

        self.data_dir = data_dir
        self.chlg_imgs_dir = os.path.join(self.data_dir, "fcn_mix_weight")

        ClassificationDataset._singleton_instance = self


def get_ds() -> ClassificationDataset:
    return ClassificationDataset.get_instance()


dc_imgs_dir = "dc_{}"
dr_imgs_dir = "dr_{}"

dc_13k_imgs_dir = "mask_rcnn_{}"
dr_13k_imgs_dir = "segmented_nih_pills_{}"

pb_masterdata_2016 = os.path.normpath("resources/pillbox_201605.tsv")

pb_masterdata = os.path.normpath("resources/pillbox_201805.tab")


def get_image_path(row: pd.Series, check_13k: bool = False, image_width: int = 224) -> str:
    if not (check_13k and row["is_new"]):
        if row["is_ref"]:
            base_dir = os.path.join(get_ds().chlg_imgs_dir, dr_imgs_dir.format(image_width))
        else:
            base_dir = os.path.join(get_ds().chlg_imgs_dir, dc_imgs_dir.format(image_width))
    else:
        if row["is_ref"]:
            base_dir = os.path.join(get_ds().data_dir, dr_13k_imgs_dir.format(image_width))
        else:
            base_dir = os.path.join(get_ds().data_dir, dc_13k_imgs_dir.format(image_width))

    return os.path.join(base_dir, row["images"])


def add_prodlbl_id_cols(pb_data: pd.DataFrame) -> None:
    split_lblprod_code = pb_data.product_code.apply(lambda x: x.split("-"))

    pb_data["label_code_id"] = split_lblprod_code.apply(lambda x: int(x[0]))
    pb_data["prod_code_id"] = split_lblprod_code.apply(
        lambda x: int(x[1][1:]) if x[1][0] == "N" else int(x[1])
    )


app_col_list = [
    "splimprint",
    "splshape_text",
    "splcolor_text",
]

relev_col_list = [
    "rxstring_new",
    "splimprint",
    "splshape_text",
    "splcolor_text",
    "product_code",
]


def load_core_pb_masterdata() -> pd.DataFrame:
    raw_pillbox_masterdata = pd.read_csv(
        os.path.join(get_ds().data_dir, pb_masterdata_2016),
        encoding="cp1252",
        delimiter="\t",
        dtype={"ndc9": str, "splimage": str},
        parse_dates=["created_at", "updated_at"],
    )
    core_pb_df = raw_pillbox_masterdata[relev_col_list].dropna().copy()
    core_pb_df.drop_duplicates(inplace=True)

    add_prodlbl_id_cols(core_pb_df)
    return core_pb_df


def load_raw_pb_masterdata_201805() -> pd.DataFrame:
    raw_pillbox_masterdata = pd.read_csv(
        os.path.join(get_ds().data_dir, pb_masterdata),
        delimiter="\t",
        dtype={"NDC9": str, "image_id": str, "RXCUI": str},
    )

    return raw_pillbox_masterdata


def load_core_pb_masterdata_201805(
    remove_all_dups: bool = True, shape_only: Optional[str] = None
) -> pd.DataFrame:
    new_relev_col_list = [
        "RXSTRING",
        "SPLIMPRINT",
        "SPLSHAPE",
        "SPLCOLOR",
        "PRODUCT_CODE",
        "RXCUI",
    ]

    raw_pillbox_masterdata = load_raw_pb_masterdata_201805()

    raw_pillbox_masterdata = raw_pillbox_masterdata[new_relev_col_list]

    if shape_only is not None:
        raw_pillbox_masterdata = raw_pillbox_masterdata[
            raw_pillbox_masterdata.SPLSHAPE == shape_only
        ]

    core_pb_df = raw_pillbox_masterdata.dropna(subset=["RXSTRING", "SPLIMPRINT", "PRODUCT_CODE"])

    if remove_all_dups:
        core_pb_df = core_pb_df.loc[
            ~core_pb_df.duplicated(subset=["PRODUCT_CODE"], keep=False)
        ]
    else:
        core_pb_df = core_pb_df.loc[
            ~core_pb_df.duplicated(subset=["PRODUCT_CODE"], keep="first")
        ]

    renamed_cols = copy.copy(relev_col_list)
    renamed_cols.append("rxcui")
    core_pb_df.columns = renamed_cols

    core_pb_df = core_pb_df.copy().reset_index()
    add_prodlbl_id_cols(core_pb_df)

    return core_pb_df


def add_app_hash_id(pb_df: pd.DataFrame) -> None:
    from hashlib import sha224

    pb_df["app_hash_id"] = pb_df[app_col_list].apply(
        lambda x: sha224("".join(x).encode()).hexdigest(), axis=1
    )


def add_label_prod_code(pd_df: pd.DataFrame) -> None:
    pd_df["label_prod_code"] = (
        pd_df["label_code_id"].map(str) + "-" + pd_df["prod_code_id"].map(str)
    )
