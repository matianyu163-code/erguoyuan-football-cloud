"""Chronological, label-availability-safe ML train/validation/test partitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from erguoyuan_football.contracts.common import utc
from erguoyuan_football.ml.dataset_builder import make_dataset
from erguoyuan_football.ml.schemas import MLDataset, MLTrainingRow


@dataclass(frozen=True)
class MLTimeSplit:
    train: MLDataset
    validation: MLDataset
    test: MLDataset
    frozen_test_hash: str

    def assert_test_unchanged(self) -> None:
        if self.test.manifest.data_hash != self.frozen_test_hash:
            raise ValueError("FINAL_TEST_SET_CHANGED")


class MLWalkForwardSplit:
    """Build expanding chronological windows; never randomize football observations."""

    @staticmethod
    def explicit(dataset: MLDataset, *, train_end: datetime, validation_end: datetime,
                 test_end: datetime) -> MLTimeSplit:
        train_at, validation_at, test_at = utc(train_end), utc(validation_end), utc(test_end)
        if not train_at < validation_at < test_at:
            raise ValueError("INVALID_TIME_SPLIT_BOUNDARIES")
        train = tuple(row for row in dataset.rows if row.vector.kickoff_time < train_at)
        validation = tuple(row for row in dataset.rows
                           if train_at <= row.vector.kickoff_time < validation_at)
        test = tuple(row for row in dataset.rows
                     if validation_at <= row.vector.kickoff_time < test_at)
        return MLWalkForwardSplit._assemble(dataset, train, validation, test)

    @staticmethod
    def rolling(dataset: MLDataset, *, initial_train: int, validation_size: int,
                test_size: int, step: int) -> tuple[MLTimeSplit, ...]:
        if min(initial_train, validation_size, test_size, step) < 1:
            raise ValueError("ML_WINDOW_SIZES_MUST_BE_POSITIVE")
        rows = dataset.rows
        windows = []
        for end in range(initial_train, len(rows) - validation_size - test_size + 1, step):
            candidate = (rows[:end], rows[end:end + validation_size],
                         rows[end + validation_size:end + validation_size + test_size])
            try:
                windows.append(MLWalkForwardSplit._assemble(dataset, *candidate))
            except ValueError as error:
                if str(error) not in {"TRAIN_LABEL_UNAVAILABLE_AT_VALIDATION",
                                      "VALIDATION_LABEL_UNAVAILABLE_AT_TEST",
                                      "OVERLAPPING_TIME_SPLIT"}:
                    raise
        return tuple(windows)

    @staticmethod
    def _assemble(dataset: MLDataset, train: tuple[MLTrainingRow, ...],
                  validation: tuple[MLTrainingRow, ...], test: tuple[MLTrainingRow, ...]) -> MLTimeSplit:
        if not train or not validation or not test:
            raise ValueError("EMPTY_ML_TIME_PARTITION")
        if not max(row.vector.kickoff_time for row in train) < min(
                row.vector.kickoff_time for row in validation) or not max(
                row.vector.kickoff_time for row in validation) < min(
                row.vector.kickoff_time for row in test):
            raise ValueError("OVERLAPPING_TIME_SPLIT")
        first_validation = min(row.vector.prediction_time for row in validation)
        first_test = min(row.vector.prediction_time for row in test)
        if max(row.label_available_at for row in train) > first_validation:
            raise ValueError("TRAIN_LABEL_UNAVAILABLE_AT_VALIDATION")
        if max(row.label_available_at for row in validation) > first_test:
            raise ValueError("VALIDATION_LABEL_UNAVAILABLE_AT_TEST")
        kind = dataset.dataset_kind
        train_set = make_dataset(train, dataset.feature_schema, dataset_kind=kind)
        validation_set = make_dataset(validation, dataset.feature_schema, dataset_kind=kind)
        test_set = make_dataset(test, dataset.feature_schema, dataset_kind=kind)
        return MLTimeSplit(train=train_set, validation=validation_set, test=test_set,
                           frozen_test_hash=test_set.manifest.data_hash)
