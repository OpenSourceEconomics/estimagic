import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest
from optimagic.logging.sqlite import (
    IterationStore,
    SQLiteConfig,
    StepStore,
)
from optimagic.logging.types import (
    CriterionEvaluationResult,
    StepResult,
    StepStatus,
    StepType,
)
from sqlalchemy import inspect


class TestIterationStore:
    @pytest.fixture
    def store(self, tmp_path):
        """Fixture to set up the IterationStore."""
        return IterationStore(SQLiteConfig(tmp_path / "test.db"))

    @staticmethod
    def create_test_point(i: int):
        return CriterionEvaluationResult(
            params=np.array([i, i + 1]),
            internal_derivative=None,
            timestamp=123456.0 + i,
            exceptions=None,
            valid=True,
            hash=f"abc{i}",
            value=0.5 + i,
            step=i,
            criterion_eval=None,
        )

    def test_table_creation(self, store):
        """Test that the IterationStore table is created properly."""
        assert store.table_name in inspect(store.engine).get_table_names()

    def test_insert_and_query(self, store):
        """Test inserting and querying data in the IterationStore."""
        result = self.create_test_point(2456)
        store.insert(result)
        queried_result = store.select(1)[0]
        assert queried_result is not None
        assert queried_result.value == result.value

    def test_update(self, store):
        """Test updating an entry in the IterationStore."""
        # Insert initial data
        result = self.create_test_point(568)
        store.insert(result)
        queried_result = store.select(1)[0]

        # Update the value
        updated_result = CriterionEvaluationResult(
            params=queried_result.params,
            internal_derivative=queried_result.internal_derivative,
            timestamp=queried_result.timestamp,
            exceptions=queried_result.exceptions,
            valid=queried_result.valid,
            hash=queried_result.hash,
            value=1.0,  # New value
            step=queried_result.step,
            criterion_eval=queried_result.criterion_eval,
        )
        store.update(key=1, value=updated_result)

        # Verify the update
        updated_entry = store.select(1)[0]
        assert updated_entry is not None
        assert updated_entry.value == 1.0

        store.update(key=1, value={"step": 34})
        updated_entry = store.select(1)[0]
        assert updated_entry is not None
        assert updated_entry.step == 34.0

    def test_serialization(self, store):
        """Test the serialization and deserialization of the IterationStore."""
        pickled_store = pickle.dumps(store)
        unpickled_store = pickle.loads(pickled_store)
        assert store.table_name == unpickled_store.table_name
        assert store.table_name in inspect(unpickled_store.engine).get_table_names()

    @pytest.mark.parametrize(
        "executor_factory",
        [
            lambda: ThreadPoolExecutor(max_workers=10),
            lambda: ProcessPoolExecutor(max_workers=10),
        ],
        ids=["threads", "processes"],
    )
    def test_parallel_insert(self, store, executor_factory):
        """Test multithreaded writing and reading in the IterationStore."""

        with executor_factory() as executor:
            # Insert data concurrently
            to_insert = list(map(self.create_test_point, range(10)))
            futures = [executor.submit(store.insert, item) for item in to_insert]
            for future in futures:
                future.result()

        result = store.select()

        assert [row.rowid for row in result] == list(range(1, 11))
        assert set([row.step for row in result]) == set(range(10))

        result_last = store.select_last_rows(5)
        assert len(result_last) == 5

    @pytest.mark.parametrize(
        "executor_factory",
        [
            lambda: ThreadPoolExecutor(max_workers=10),
            lambda: ProcessPoolExecutor(max_workers=10),
        ],
        ids=["threads", "processes"],
    )
    def test_parallel_update(self, store, executor_factory):
        """Test multithreaded writing and reading in the IterationStore."""

        with executor_factory() as executor:
            # Insert data concurrently
            to_insert = list(map(self.create_test_point, range(10)))
            futures = [executor.submit(store.insert, item) for item in to_insert]
            for future in futures:
                future.result()

        with executor_factory() as executor:
            # Update data concurrently
            to_update = [(2, {"value": 100}), (2, {"step": 200})]
            futures = [executor.submit(store.update, *item) for item in to_update]
            for future in futures:
                future.result()

        result = store.select(2)[0]
        assert result.value == 100
        assert result.step == 200


class TestStepStore:
    @pytest.fixture
    def store(self, tmp_path):
        """Fixture to set up the IterationStore."""
        return StepStore(SQLiteConfig(tmp_path / "test.db"))

    @staticmethod
    def create_test_point(i: int):
        return StepResult(
            f"random_{i}", StepType.OPTIMIZATION, StepStatus.RUNNING, n_iterations=i
        )

    def test_table_creation(self, store):
        """Test that the IterationStore table is created properly."""
        assert store.table_name in inspect(store.engine).get_table_names()

    def test_insert_and_query(self, store):
        """Test inserting and querying data in the IterationStore."""
        result = self.create_test_point(2456)
        store.insert(result)
        queried_result = store.select(1)[0]
        assert queried_result is not None
        assert queried_result.n_iterations == result.n_iterations

    def test_insert_string(self, store):
        result = StepResult("strings", "optimization", "running", n_iterations=1)
        store.insert(result)
        queried_result = store.select(1)[0]
        assert queried_result is not None
        assert queried_result.status is StepStatus.RUNNING
        assert queried_result.type is StepType.OPTIMIZATION

    def test_update(self, store):
        """Test updating an entry in the IterationStore."""
        # Insert initial data
        result = self.create_test_point(568)
        store.insert(result)
        queried_result = store.select(1)[0]

        # Update the value
        updated_result = StepResult(
            queried_result.name,
            queried_result.type,
            queried_result.status,
            n_iterations=50,
        )
        store.update(key=1, value=updated_result)

        # Verify the update
        updated_entry = store.select(1)[0]
        assert updated_entry is not None
        assert updated_entry.n_iterations == 50

        store.update(key=1, value={"n_iterations": 34})
        updated_entry = store.select(1)[0]
        assert updated_entry is not None
        assert updated_entry.n_iterations == 34

    def test_serialization(self, store):
        """Test the serialization and deserialization of the IterationStore."""
        pickled_store = pickle.dumps(store)
        unpickled_store = pickle.loads(pickled_store)
        assert store.table_name == unpickled_store.table_name
        assert store.table_name in inspect(unpickled_store.engine).get_table_names()

    @pytest.mark.parametrize(
        "executor_factory",
        [
            lambda: ThreadPoolExecutor(max_workers=10),
            lambda: ProcessPoolExecutor(max_workers=10),
        ],
        ids=["threads", "processes"],
    )
    def test_parallel_insert(self, store, executor_factory):
        """Test multithreaded writing and reading in the IterationStore."""

        with executor_factory() as executor:
            # Insert data concurrently
            to_insert = list(map(self.create_test_point, range(10)))
            futures = [executor.submit(store.insert, item) for item in to_insert]
            for future in futures:
                future.result()

        result = store.select()

        assert [row.rowid for row in result] == list(range(1, 11))
        assert set([row.n_iterations for row in result]) == set(range(10))

        result_last = store.select_last_rows(5)
        assert len(result_last) == 5

    @pytest.mark.parametrize(
        "executor_factory",
        [
            lambda: ThreadPoolExecutor(max_workers=10),
            lambda: ProcessPoolExecutor(max_workers=10),
        ],
        ids=["threads", "processes"],
    )
    def test_parallel_update(self, store, executor_factory):
        """Test multithreaded writing and reading in the IterationStore."""

        with executor_factory() as executor:
            # Insert data concurrently
            to_insert = list(map(self.create_test_point, range(10)))
            futures = [executor.submit(store.insert, item) for item in to_insert]
            for future in futures:
                future.result()

        with executor_factory() as executor:
            # Update data concurrently
            to_update = [
                (2, {"status": StepStatus.COMPLETE}),
                (2, {"n_iterations": 200}),
            ]
            futures = [executor.submit(store.update, *item) for item in to_update]
            for future in futures:
                future.result()

        result = store.select(2)[0]
        assert result.status == StepStatus.COMPLETE
        assert result.n_iterations == 200
