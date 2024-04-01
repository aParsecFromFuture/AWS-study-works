import json
from deepchecks.tabular.checks import *
from deepchecks.tabular import Dataset


REPORT_TYPE = [
    "Is Single Value",
    "Percent Of Nulls",
    "Feature Label Correlation",
    "Mixed Data Types",
    "Class Imbalance",
    "Feature Drift",
    "Multivariate Drift",
    "New Category Train Test",
    "Train Test Samples Mix",
]


def is_single_value(ds):
    ds = Dataset(ds, label=ds.columns[-1])
    check = IsSingleValue()
    check.add_condition_not_single_value()
    result = check.run(ds)
    return json.loads(result.to_json())


def percent_of_nulls(ds):
    ds = Dataset(ds, label=ds.columns[-1])
    check = PercentOfNulls()
    check.add_condition_percent_of_nulls_not_greater_than(0.95)
    result = check.run(ds)
    return json.loads(result.to_json())


def feature_label_correlation(ds):
    ds = Dataset(ds, label=ds.columns[-1])
    check = FeatureLabelCorrelation()
    check.add_condition_feature_pps_less_than(0.8)
    result = check.run(ds)
    return json.loads(result.to_json())


def mixed_data_types(ds):
    ds = Dataset(ds, label=ds.columns[-1])
    check = MixedDataTypes()
    check.add_condition_rare_type_ratio_not_in_range()
    result = check.run(ds)
    return json.loads(result.to_json())


def class_imbalance(ds):
    ds = Dataset(ds, label=ds.columns[-1])
    check = ClassImbalance()
    check.add_condition_class_ratio_less_than(0.1)
    result = check.run(ds)
    return json.loads(result.to_json())


def multivariate_drift(ds_train, ds_test):
    ds_train = Dataset(ds_train, label=ds_train.columns[-1])
    check = MultivariateDrift()
    check.add_condition_overall_drift_value_less_than(0.25)
    result = check.run(ds_train, ds_test)
    return json.loads(result.to_json())


def feature_drift(ds_train, ds_test):
    ds_train = Dataset(ds_train, label=ds_train.columns[-1])
    ds_test = Dataset(ds_test, label=None)
    check = FeatureDrift()
    check.add_condition_drift_score_less_than(0.2, 0.2)
    result = check.run(ds_train, ds_test)
    return json.loads(result.to_json())


def new_category_train_test(ds_train, ds_test):
    ds_train = Dataset(ds_train, label=ds_train.columns[-1])
    ds_test = Dataset(ds_test, label=None)
    check = NewCategoryTrainTest()
    check.add_condition_new_categories_less_or_equal()
    result = check.run(ds_train, ds_test)
    return json.loads(result.to_json())


def train_test_samples_mix(ds_train, ds_test):
    ds_train = Dataset(ds_train, label=ds_train.columns[-1])
    ds_test = Dataset(ds_test, label=None)
    check = TrainTestSamplesMix()
    check.add_condition_duplicates_ratio_less_or_equal(0.0)
    result = check.run(ds_train, ds_test)
    return json.loads(result.to_json())
