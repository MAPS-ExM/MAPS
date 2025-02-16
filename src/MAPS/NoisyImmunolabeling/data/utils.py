from typing import Union
from dataclasses import dataclass


@dataclass
class Sample:
    file: str
    path_nhs: str
    path_pred: str

    def get_nhs_name(self):
        return self.file[: self.file.find("_nhs")] + ".tif"


@dataclass
class FileRecord:
    file_name: str
    path_base: str
    file_outline_pred: str
    path_outline_pred: str
    file_inner_pred: str
    path_inner_pred: str
    nhs_lower: Union[int, float]
    nhs_upper: Union[int, float]
