# Authors: The EEGDash contributors.
# License: GNU General Public License
# Copyright the EEGDash contributors.

"""Configuration constants and mappings for EEGDash.

This module contains global configuration settings, allowed query fields, and mapping
constants used throughout the EEGDash package. It defines the interface between EEGDash
releases and OpenNeuro dataset identifiers, as well as validation rules for database queries.
"""

__all__ = [
    "config",
    "ALLOWED_QUERY_FIELDS",
    "RELEASE_TO_OPENNEURO_DATASET_MAP",
    "SUBJECT_MINI_RELEASE_MAP",
]

ALLOWED_QUERY_FIELDS = {
    "data_name",
    "dataset",
    "subject",
    "task",
    "session",
    "run",
    "modality",
    "sampling_frequency",
    "nchans",
    "ntimes",
}
"""set: A set of field names that are permitted in database queries constructed
via :func:`~eegdash.api.EEGDash.find` with keyword arguments."""

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
"""dict: A mapping from Healthy Brain Network (HBN) release identifiers (e.g., "R11")
to their corresponding OpenNeuro dataset identifiers (e.g., "ds005516")."""

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
"""dict: A mapping from HBN release identifiers to a list of subject IDs.
This is used to select a small, representative subset of subjects for creating
"mini" datasets for testing and demonstration purposes."""

config = {
    "required_fields": ["data_name"],
    # Default set of user-facing primary record attributes expected in the database. Records
    # where any of these are missing will be loaded with the respective attribute set to None.
    # Additional fields may be returned if they are present in the database, notably bidsdependencies.
    "attributes": {
        "data_name": "str",
        "dataset": "str",
        "bidspath": "str",
        "subject": "str",
        "task": "str",
        "session": "str",
        "run": "str",
        "sampling_frequency": "float",
        "modality": "str",
        "nchans": "int",
        "ntimes": "int",  # note: this is really the number of seconds in the data, rounded down
    },
    # queryable descriptive fields for a given recording
    "description_fields": ["subject", "session", "run", "task", "age", "gender", "sex"],
    # list of filenames that may be present in the BIDS dataset directory that are used
    # to load and interpret a given BIDS recording.
    "bids_dependencies_files": [
        "dataset_description.json",
        "participants.tsv",
        "events.tsv",
        "events.json",
        "eeg.json",
        "electrodes.tsv",
        "channels.tsv",
        "coordsystem.json",
    ],
    "accepted_query_fields": ["data_name", "dataset"],
}
"""dict: A global configuration dictionary for the EEGDash package.

Keys
----
required_fields : list
    Fields that must be present in every database record.
attributes : dict
    A schema defining the expected primary attributes and their types for a
    database record.
description_fields : list
    A list of fields considered to be descriptive metadata for a recording,
    which can be used for filtering and display.
bids_dependencies_files : list
    A list of BIDS metadata filenames that are relevant for interpreting an
    EEG recording.
accepted_query_fields : list
    Fields that are accepted for lightweight existence checks in the database.
"""
