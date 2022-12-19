# python peripherals
import os
from multiprocessing import Process, Queue
from pathlib import Path
import random
from abc import ABC, abstractmethod
from typing import List, Optional, cast
import traceback

# numpy
import numpy

# deep-signature
from deep_signature.core.base import LoggerObject


# =================================================
# ParallelProcessorBase Class
# =================================================
class ParallelProcessorBase(ABC, LoggerObject):
    def __init__(self, log_dir_path: Path, num_workers: int, **kw: object):
        self._num_workers = num_workers
        self._workers = []
        super().__init__(log_dir_path=log_dir_path, **kw)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_workers']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def start(self):
        num_workers_digits = len(str(self._num_workers))

        self._logger.info(msg='Running Pre Start')
        self._pre_start()

        self._workers = [Process(target=self._worker_func, args=tuple([worker_id],)) for worker_id in range(self._num_workers)]

        print('')
        for i, worker in enumerate(self._workers):
            worker.start()
            print(f'\rWorker Started {i+1:{" "}{"<"}{num_workers_digits}} / {self._num_workers:{" "}{">"}{num_workers_digits}}', end='')
        print('')

        self._logger.info(msg='Running Post Start')
        self._post_start()

    def join(self):
        self._logger.info(msg='Running Pre-Join')
        self._pre_join()

        self._logger.info(msg='Joining processes')
        for worker in self._workers:
            worker.join()

        self._logger.info(msg='Running Post-Join')
        self._post_join()

    @abstractmethod
    def _pre_start(self):
        pass

    @abstractmethod
    def _post_start(self):
        pass

    @abstractmethod
    def _pre_join(self):
        pass

    @abstractmethod
    def _post_join(self):
        pass

    @abstractmethod
    def _worker_func(self, worker_id: int):
        pass


# =================================================
# ParallelProcessorTask Class
# =================================================
class ParallelProcessingTask(ABC):
    def __init__(self):
        super().__init__()

    def process(self):
        self._pre_process()
        self._process()
        self._post_process()

    @abstractmethod
    def _pre_process(self):
        pass

    @abstractmethod
    def _process(self):
        pass

    @abstractmethod
    def _post_process(self):
        pass


# =================================================
# TaskParallelProcessor Class
# =================================================
class TaskParallelProcessor(ParallelProcessorBase):
    def __init__(self, log_dir_path: Path, num_workers: int, max_tasks: Optional[int] = None, **kw: object):
        super().__init__(log_dir_path=log_dir_path, num_workers=num_workers, **kw)
        self._tasks_queue = Queue()
        self._completed_tasks_queue = Queue()
        self._max_tasks = max_tasks
        tasks = self._generate_tasks()
        self._tasks = random.sample(tasks, max_tasks)
        self._completed_tasks = []

    @property
    def tasks_count(self) -> int:
        return len(self._tasks)

    def _pre_start(self):
        for task in self._tasks:
            self._tasks_queue.put(obj=task)

        for _ in range(self._num_workers):
            self._tasks_queue.put(obj=None)

    def _post_start(self):
        last_remaining_tasks_count = numpy.inf
        tasks_count_digits = len(str(self.tasks_count))
        while True:
            remaining_tasks_count = self._tasks_queue.qsize()
            if last_remaining_tasks_count > remaining_tasks_count:
                print(f'\rRemaining Tasks {numpy.maximum(remaining_tasks_count - self._num_workers, 0):{" "}{"<"}{tasks_count_digits}} / {self.tasks_count:{" "}{">"}{tasks_count_digits}}', end='')
                last_remaining_tasks_count = remaining_tasks_count

            if remaining_tasks_count == 0:
                break

        print('')

        self._logger.info(msg='Draining Queue')
        sentinels_count = 0
        while True:
            completed_task = self._completed_tasks_queue.get()
            if completed_task is None:
                sentinels_count = sentinels_count + 1
            else:
                self._completed_tasks.append(completed_task)

            if sentinels_count == self._num_workers:
                break

    @abstractmethod
    def _pre_join(self):
        pass

    @abstractmethod
    def _post_join(self):
        pass

    @abstractmethod
    def _generate_tasks(self) -> List[ParallelProcessingTask]:
        pass

    def _worker_func(self, worker_id: int):
        while True:
            task = cast(typ=ParallelProcessingTask, val=self._tasks_queue.get())
            if task is None:
                self._completed_tasks_queue.put(None)
                return

            try:
                task.process()
            except:
                print()
                traceback.print_exc()

            self._completed_tasks_queue.put(task)
