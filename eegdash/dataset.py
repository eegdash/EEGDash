from pathlib import Path

from .api import EEGDashDataset
from .registry import register_openneuro_datasets

RELEASE_TO_OPENNEURO_DATASET_MAP = {
    "R11": "ds005516",
    "R10": "ds005515",
    "R9": "ds005514",
    "R8": "ds005512",
    "R7": "ds005511",
    "R6": "ds005510",
    "R4": "ds005508",
    "R5": "ds005509",
    "R3": "ds005507",
    "R2": "ds005506",
    "R1": "ds005505",
}

SUBJECT_MINI_RELEASE_MAP = {
    "R11": [
        "sub-NDARAB678VYW",
        "sub-NDARAG788YV9",
        "sub-NDARAM946HJE",
        "sub-NDARAY977BZT",
        "sub-NDARAZ532KK0",
        "sub-NDARCE912ZXW",
        "sub-NDARCM214WFE",
        "sub-NDARDL033XRG",
        "sub-NDARDT889RT9",
        "sub-NDARDZ794ZVP",
        "sub-NDAREV869CPW",
        "sub-NDARFN221WW5",
        "sub-NDARFV289RKB",
        "sub-NDARFY623ZTE",
        "sub-NDARGA890MKA",
        "sub-NDARHN206XY3",
        "sub-NDARHP518FUR",
        "sub-NDARJL292RYV",
        "sub-NDARKM199DXW",
        "sub-NDARKW236TN7",
    ],
    "R10": [
        "sub-NDARAC589YMB",
        "sub-NDARAC853CR6",
        "sub-NDARAH239PGG",
        "sub-NDARAL897CYV",
        "sub-NDARAN160GUF",
        "sub-NDARAP049KXJ",
        "sub-NDARAP457WB5",
        "sub-NDARAW216PM7",
        "sub-NDARBA004KBT",
        "sub-NDARBD328NUQ",
        "sub-NDARBF042LDM",
        "sub-NDARBH019KPD",
        "sub-NDARBH728DFK",
        "sub-NDARBM370JCB",
        "sub-NDARBU183TDJ",
        "sub-NDARBW971DCW",
        "sub-NDARBZ444ZHK",
        "sub-NDARCC620ZFT",
        "sub-NDARCD182XT1",
        "sub-NDARCK113CJM",
    ],
    "R9": [
        "sub-NDARAC589YMB",
        "sub-NDARAC853CR6",
        "sub-NDARAH239PGG",
        "sub-NDARAL897CYV",
        "sub-NDARAN160GUF",
        "sub-NDARAP049KXJ",
        "sub-NDARAP457WB5",
        "sub-NDARAW216PM7",
        "sub-NDARBA004KBT",
        "sub-NDARBD328NUQ",
        "sub-NDARBF042LDM",
        "sub-NDARBH019KPD",
        "sub-NDARBH728DFK",
        "sub-NDARBM370JCB",
        "sub-NDARBU183TDJ",
        "sub-NDARBW971DCW",
        "sub-NDARBZ444ZHK",
        "sub-NDARCC620ZFT",
        "sub-NDARCD182XT1",
        "sub-NDARCK113CJM",
    ],
    "R8": [
        "sub-NDARAB514MAJ",
        "sub-NDARAD571FLB",
        "sub-NDARAF003VCL",
        "sub-NDARAG191AE8",
        "sub-NDARAJ977PRJ",
        "sub-NDARAP912JK3",
        "sub-NDARAV454VF0",
        "sub-NDARAY298THW",
        "sub-NDARBJ375VP4",
        "sub-NDARBT436PMT",
        "sub-NDARBV630BK6",
        "sub-NDARCB627KDN",
        "sub-NDARCC059WTH",
        "sub-NDARCM953HKD",
        "sub-NDARCN681CXW",
        "sub-NDARCT889DMB",
        "sub-NDARDJ204EPU",
        "sub-NDARDJ544BU5",
        "sub-NDARDP292DVC",
        "sub-NDARDW178AC6",
    ],
    "R7": [
        "sub-NDARAY475AKD",
        "sub-NDARBW026UGE",
        "sub-NDARCK162REX",
        "sub-NDARCK481KRH",
        "sub-NDARCV378MMX",
        "sub-NDARCX462NVA",
        "sub-NDARDJ970ELG",
        "sub-NDARDU617ZW1",
        "sub-NDAREM609ZXW",
        "sub-NDAREW074ZM2",
        "sub-NDARFE555KXB",
        "sub-NDARFT176NJP",
        "sub-NDARGK442YHH",
        "sub-NDARGM439FZD",
        "sub-NDARGT634DUJ",
        "sub-NDARHE283KZN",
        "sub-NDARHG260BM9",
        "sub-NDARHL684WYU",
        "sub-NDARHN224TPA",
        "sub-NDARHP841RMR",
    ],
    "R6": [
        "sub-NDARAD224CRB",
        "sub-NDARAE301XTM",
        "sub-NDARAT680GJA",
        "sub-NDARCA578CEB",
        "sub-NDARDZ147ETZ",
        "sub-NDARFL793LDE",
        "sub-NDARFX710UZA",
        "sub-NDARGE994BMX",
        "sub-NDARGP191YHN",
        "sub-NDARGV436PFT",
        "sub-NDARHF545HFW",
        "sub-NDARHP039DBU",
        "sub-NDARHT774ZK1",
        "sub-NDARJA830BYV",
        "sub-NDARKB614KGY",
        "sub-NDARKM250ET5",
        "sub-NDARKZ085UKQ",
        "sub-NDARLB581AXF",
        "sub-NDARNJ899HW7",
        "sub-NDARRZ606EDP",
    ],
    "R4": [
        "sub-NDARAC350BZ0",
        "sub-NDARAD615WLJ",
        "sub-NDARAG584XLU",
        "sub-NDARAH503YG1",
        "sub-NDARAX272ZJL",
        "sub-NDARAY461TZZ",
        "sub-NDARBC734UVY",
        "sub-NDARBL444FBA",
        "sub-NDARBT640EBN",
        "sub-NDARBU098PJT",
        "sub-NDARBU928LV0",
        "sub-NDARBV059CGE",
        "sub-NDARCG037CX4",
        "sub-NDARCG947ZC0",
        "sub-NDARCH001CN2",
        "sub-NDARCU001ZN7",
        "sub-NDARCW497XW2",
        "sub-NDARCX053GU5",
        "sub-NDARDF568GL5",
        "sub-NDARDJ092YKH",
    ],
    "R5": [
        "sub-NDARAH793FBF",
        "sub-NDARAJ689BVN",
        "sub-NDARAP785CTE",
        "sub-NDARAU708TL8",
        "sub-NDARBE091BGD",
        "sub-NDARBE103DHM",
        "sub-NDARBF851NH6",
        "sub-NDARBH228RDW",
        "sub-NDARBJ674TVU",
        "sub-NDARBM433VER",
        "sub-NDARCA740UC8",
        "sub-NDARCU633GCZ",
        "sub-NDARCU736GZ1",
        "sub-NDARCU744XWL",
        "sub-NDARDC843HHM",
        "sub-NDARDH086ZKK",
        "sub-NDARDL305BT8",
        "sub-NDARDU853XZ6",
        "sub-NDARDV245WJG",
        "sub-NDAREC480KFA",
    ],
    "R3": [
        "sub-NDARAA948VFH",
        "sub-NDARAD774HAZ",
        "sub-NDARAE828CML",
        "sub-NDARAG340ERT",
        "sub-NDARBA839HLG",
        "sub-NDARBE641DGZ",
        "sub-NDARBG574KF4",
        "sub-NDARBM642JFT",
        "sub-NDARCL016NHB",
        "sub-NDARCV944JA6",
        "sub-NDARCY178KJP",
        "sub-NDARDY150ZP9",
        "sub-NDAREC542MH3",
        "sub-NDAREK549XUQ",
        "sub-NDAREM887YY8",
        "sub-NDARFA815FXE",
        "sub-NDARFF644ZGD",
        "sub-NDARFV557XAA",
        "sub-NDARFV780ABD",
        "sub-NDARGB102NWJ",
    ],
    "R2": [
        "sub-NDARAB793GL3",
        "sub-NDARAM675UR8",
        "sub-NDARBM839WR5",
        "sub-NDARBU730PN8",
        "sub-NDARCT974NAJ",
        "sub-NDARCW933FD5",
        "sub-NDARCZ770BRG",
        "sub-NDARDW741HCF",
        "sub-NDARDZ058NZN",
        "sub-NDAREC377AU2",
        "sub-NDAREM500WWH",
        "sub-NDAREV527ZRF",
        "sub-NDAREV601CE7",
        "sub-NDARFF070XHV",
        "sub-NDARFR108JNB",
        "sub-NDARFT305CG1",
        "sub-NDARGA056TMW",
        "sub-NDARGH775KF5",
        "sub-NDARGJ878ZP4",
        "sub-NDARHA387FPM",
    ],
    "R1": [
        "sub-NDARAC904DMU",
        "sub-NDARAM704GKZ",
        "sub-NDARAP359UM6",
        "sub-NDARBD879MBX",
        "sub-NDARBH024NH2",
        "sub-NDARBK082PDD",
        "sub-NDARCA153NKE",
        "sub-NDARCE721YB5",
        "sub-NDARCJ594BWQ",
        "sub-NDARCN669XPR",
        "sub-NDARCW094JCG",
        "sub-NDARCZ947WU5",
        "sub-NDARDH670PXH",
        "sub-NDARDL511UND",
        "sub-NDARDU986RBM",
        "sub-NDAREM731BYM",
        "sub-NDAREN519BLJ",
        "sub-NDARFK610GY5",
        "sub-NDARFT581ZW5",
        "sub-NDARFW972KFQ",
    ],
}


class EEGChallengeDataset(EEGDashDataset):
    def __init__(
        self,
        release: str | list[str],
        cache_dir: str,
        mini: bool = True,
        query: dict | None = None,
        s3_bucket: str | None = "s3://nmdatasets/NeurIPS25",
        **kwargs,
    ):
        """Create a new EEGDashDataset from a given query or local BIDS dataset directory
        and dataset name. An EEGDashDataset is pooled collection of EEGDashBaseDataset
        instances (individual recordings) and is a subclass of braindecode's BaseConcatDataset.

        Parameters
        ----------
        release: str
            Release name. Can be one of ["R1", ..., "R11"]
        mini: bool, default True
            Whether to use the mini-release version of the dataset. It is recommended
            to use the mini version for faster training and evaluation.
        query : dict | None
            Optionally a dictionary that specifies a query to be executed,
            in addition to the dataset (automatically inferred from the release argument).
            See EEGDash.find() for details on the query format.
        cache_dir : str
            A directory where the dataset will be cached locally.
        s3_bucket : str | None
            An optional S3 bucket URI to use instead of the
            default OpenNeuro bucket for loading data files.
        kwargs : dict
            Additional keyword arguments to be passed to the EEGDashDataset
            constructor.

        """
        self.release = release
        if release not in RELEASE_TO_OPENNEURO_DATASET_MAP:
            raise ValueError(f"Unknown release: {release}")

        dataset_parameters = []
        if isinstance(release, str):
            dataset_parameters.append(RELEASE_TO_OPENNEURO_DATASET_MAP[release])
        else:
            raise ValueError(
                f"Unknown release type: {type(release)}, the expected type is str."
            )

        if "dataset" in query:
            raise ValueError(
                "Query using the parameters `dataset` with the class EEGChallengeDataset is not possible"
                "Please use the release argument instead, or the object EEGDashDataset instead."
            )

        if mini:
            if "subject" in query:
                raise ValueError(
                    "Query using the parameters `subject` with the class EEGChallengeDataset and `mini==True` is not possible"
                    "Please don't use the `subject` selection twice."
                    "Set `mini=False` to use the `subject` selection."
                )
            kwargs["subject"] = SUBJECT_MINI_RELEASE_MAP[release]
            s3_bucket = f"{s3_bucket}/R{release}_mini_L100_bdf"
        else:
            s3_bucket = f"{s3_bucket}/R{release}_L100_bdf"

        super().__init__(
            dataset=dataset_parameters,
            query=query,
            cache_dir=cache_dir,
            s3_bucket=s3_bucket,
            **kwargs,
        )


registered_classes = register_openneuro_datasets(
    summary_file=Path(__file__).with_name("dataset_summary.csv"),
    base_class=EEGDashDataset,
    namespace=globals(),
)


__all__ = ["EEGChallengeDataset"] + list(registered_classes.keys())
