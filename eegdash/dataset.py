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
        "NDARAB678VYW",
        "NDARAG788YV9",
        "NDARAM946HJE",
        "NDARAY977BZT",
        "NDARAZ532KK0",
        "NDARCE912ZXW",
        "NDARCM214WFE",
        "NDARDL033XRG",
        "NDARDT889RT9",
        "NDARDZ794ZVP",
        "NDAREV869CPW",
        "NDARFN221WW5",
        "NDARFV289RKB",
        "NDARFY623ZTE",
        "NDARGA890MKA",
        "NDARHN206XY3",
        "NDARHP518FUR",
        "NDARJL292RYV",
        "NDARKM199DXW",
        "NDARKW236TN7",
    ],
    "R10": [
        "NDARAR935TGZ",
        "NDARAV474ADJ",
        "NDARCB869VM8",
        "NDARCJ667UPL",
        "NDARCM677TC1",
        "NDARET671FTC",
        "NDARKM061NHZ",
        "NDARLD501HDK",
        "NDARLL176DJR",
        "NDARMT791WDH",
        "NDARMW299ZAB",
        "NDARNC405WJA",
        "NDARNP962TJK",
        "NDARPB967KU7",
        "NDARRU560AGK",
        "NDARTB173LY2",
        "NDARUW377KAE",
        "NDARVH565FX9",
        "NDARVP799KGY",
        "NDARVY962GB5",
    ],
    "R9": [
        "NDARAC589YMB",
        "NDARAC853CR6",
        "NDARAH239PGG",
        "NDARAL897CYV",
        "NDARAN160GUF",
        "NDARAP049KXJ",
        "NDARAP457WB5",
        "NDARAW216PM7",
        "NDARBA004KBT",
        "NDARBD328NUQ",
        "NDARBF042LDM",
        "NDARBH019KPD",
        "NDARBH728DFK",
        "NDARBM370JCB",
        "NDARBU183TDJ",
        "NDARBW971DCW",
        "NDARBZ444ZHK",
        "NDARCC620ZFT",
        "NDARCD182XT1",
        "NDARCK113CJM",
    ],
    "R8": [
        "NDARAB514MAJ",
        "NDARAD571FLB",
        "NDARAF003VCL",
        "NDARAG191AE8",
        "NDARAJ977PRJ",
        "NDARAP912JK3",
        "NDARAV454VF0",
        "NDARAY298THW",
        "NDARBJ375VP4",
        "NDARBT436PMT",
        "NDARBV630BK6",
        "NDARCB627KDN",
        "NDARCC059WTH",
        "NDARCM953HKD",
        "NDARCN681CXW",
        "NDARCT889DMB",
        "NDARDJ204EPU",
        "NDARDJ544BU5",
        "NDARDP292DVC",
        "NDARDW178AC6",
    ],
    "R7": [
        "NDARAY475AKD",
        "NDARBW026UGE",
        "NDARCK162REX",
        "NDARCK481KRH",
        "NDARCV378MMX",
        "NDARCX462NVA",
        "NDARDJ970ELG",
        "NDARDU617ZW1",
        "NDAREM609ZXW",
        "NDAREW074ZM2",
        "NDARFE555KXB",
        "NDARFT176NJP",
        "NDARGK442YHH",
        "NDARGM439FZD",
        "NDARGT634DUJ",
        "NDARHE283KZN",
        "NDARHG260BM9",
        "NDARHL684WYU",
        "NDARHN224TPA",
        "NDARHP841RMR",
    ],
    "R6": [
        "NDARAD224CRB",
        "NDARAE301XTM",
        "NDARAT680GJA",
        "NDARCA578CEB",
        "NDARDZ147ETZ",
        "NDARFL793LDE",
        "NDARFX710UZA",
        "NDARGE994BMX",
        "NDARGP191YHN",
        "NDARGV436PFT",
        "NDARHF545HFW",
        "NDARHP039DBU",
        "NDARHT774ZK1",
        "NDARJA830BYV",
        "NDARKB614KGY",
        "NDARKM250ET5",
        "NDARKZ085UKQ",
        "NDARLB581AXF",
        "NDARNJ899HW7",
        "NDARRZ606EDP",
    ],
    "R4": [
        "NDARAC350BZ0",
        "NDARAD615WLJ",
        "NDARAG584XLU",
        "NDARAH503YG1",
        "NDARAX272ZJL",
        "NDARAY461TZZ",
        "NDARBC734UVY",
        "NDARBL444FBA",
        "NDARBT640EBN",
        "NDARBU098PJT",
        "NDARBU928LV0",
        "NDARBV059CGE",
        "NDARCG037CX4",
        "NDARCG947ZC0",
        "NDARCH001CN2",
        "NDARCU001ZN7",
        "NDARCW497XW2",
        "NDARCX053GU5",
        "NDARDF568GL5",
        "NDARDJ092YKH",
    ],
    "R5": [
        "NDARAH793FBF",
        "NDARAJ689BVN",
        "NDARAP785CTE",
        "NDARAU708TL8",
        "NDARBE091BGD",
        "NDARBE103DHM",
        "NDARBF851NH6",
        "NDARBH228RDW",
        "NDARBJ674TVU",
        "NDARBM433VER",
        "NDARCA740UC8",
        "NDARCU633GCZ",
        "NDARCU736GZ1",
        "NDARCU744XWL",
        "NDARDC843HHM",
        "NDARDH086ZKK",
        "NDARDL305BT8",
        "NDARDU853XZ6",
        "NDARDV245WJG",
        "NDAREC480KFA",
    ],
    "R3": [
        "NDARAA948VFH",
        "NDARAD774HAZ",
        "NDARAE828CML",
        "NDARAG340ERT",
        "NDARBA839HLG",
        "NDARBE641DGZ",
        "NDARBG574KF4",
        "NDARBM642JFT",
        "NDARCL016NHB",
        "NDARCV944JA6",
        "NDARCY178KJP",
        "NDARDY150ZP9",
        "NDAREC542MH3",
        "NDAREK549XUQ",
        "NDAREM887YY8",
        "NDARFA815FXE",
        "NDARFF644ZGD",
        "NDARFV557XAA",
        "NDARFV780ABD",
        "NDARGB102NWJ",
    ],
    "R2": [
        "NDARAB793GL3",
        "NDARAM675UR8",
        "NDARBM839WR5",
        "NDARBU730PN8",
        "NDARCT974NAJ",
        "NDARCW933FD5",
        "NDARCZ770BRG",
        "NDARDW741HCF",
        "NDARDZ058NZN",
        "NDAREC377AU2",
        "NDAREM500WWH",
        "NDAREV527ZRF",
        "NDAREV601CE7",
        "NDARFF070XHV",
        "NDARFR108JNB",
        "NDARFT305CG1",
        "NDARGA056TMW",
        "NDARGH775KF5",
        "NDARGJ878ZP4",
        "NDARHA387FPM",
    ],
    "R1": [
        "NDARAC904DMU",
        "NDARAM704GKZ",
        "NDARAP359UM6",
        "NDARBD879MBX",
        "NDARBH024NH2",
        "NDARBK082PDD",
        "NDARCA153NKE",
        "NDARCE721YB5",
        "NDARCJ594BWQ",
        "NDARCN669XPR",
        "NDARCW094JCG",
        "NDARCZ947WU5",
        "NDARDH670PXH",
        "NDARDL511UND",
        "NDARDU986RBM",
        "NDAREM731BYM",
        "NDAREN519BLJ",
        "NDARFK610GY5",
        "NDARFT581ZW5",
        "NDARFW972KFQ",
    ],
}


class EEGChallengeDataset(EEGDashDataset):
    def __init__(
        self,
        release: str,
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
        self.mini = mini

        if release not in RELEASE_TO_OPENNEURO_DATASET_MAP:
            raise ValueError(
                f"Unknown release: {release}, expected one of {list(RELEASE_TO_OPENNEURO_DATASET_MAP.keys())}"
            )

        dataset_parameters = []
        if isinstance(release, str):
            dataset_parameters.append(RELEASE_TO_OPENNEURO_DATASET_MAP[release])
        else:
            raise ValueError(
                f"Unknown release type: {type(release)}, the expected type is str."
            )

        if query and "dataset" in query:
            raise ValueError(
                "Query using the parameters `dataset` with the class EEGChallengeDataset is not possible."
                "Please use the release argument instead, or the object EEGDashDataset instead."
            )

        if self.mini:
            # Disallow mixing subject selection with mini=True since mini already
            # applies a predefined subject subset.
            if (query and "subject" in query) or ("subject" in kwargs):
                raise ValueError(
                    "Query using the parameters `subject` with the class EEGChallengeDataset and `mini==True` is not possible."
                    "Please don't use the `subject` selection twice."
                    "Set `mini=False` to use the `subject` selection."
                )
            kwargs["subject"] = SUBJECT_MINI_RELEASE_MAP[release]
            s3_bucket = f"{s3_bucket}/{release}_mini_L100_bdf"
        else:
            s3_bucket = f"{s3_bucket}/{release}_L100_bdf"

        super().__init__(
            dataset=RELEASE_TO_OPENNEURO_DATASET_MAP[release],
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
