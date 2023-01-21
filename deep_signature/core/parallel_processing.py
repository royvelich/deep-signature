# python peripherals
import os
from multiprocessing import Process, Queue
import queue
from pathlib import Path
import random
from abc import ABC, abstractmethod
from typing import List, Optional, cast
import traceback
from enum import Enum, auto

# torch
import torch.multiprocessing
from torch.multiprocessing import Manager, Process

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
        self._manager = Manager()
        self._namespace = self._manager.Namespace()
        super().__init__(log_dir_path=log_dir_path, **kw)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_workers']
        states_to_remove = self._get_states_to_remove()
        for state in states_to_remove:
            del d[state]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def start(self, num_workers: Optional[int] = None):
        self._num_workers = num_workers if num_workers is not None else self._num_workers
        num_workers_digits = len(str(num_workers))

        self._logger.info(msg='Running Pre Start')
        self._pre_start()

        self._workers = [Process(target=self._worker_func, args=tuple(self._get_args(),)) for _ in range(self._num_workers)]

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

    def _get_args(self) -> List[object]:
        return []

    def _get_states_to_remove(self) -> List[str]:
        return ['_manager']

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
    def _worker_func(self, *kwargs):
        pass


# =================================================
# ParallelProcessorTask Class
# =================================================
class ParallelProcessingTask(ABC):
    def __init__(self):
        super().__init__()

    def process(self, *argv):
        self._pre_process()
        self._process(*argv)
        self._post_process()

    @abstractmethod
    def _pre_process(self):
        pass

    @abstractmethod
    def _process(self, *argv):
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
        self._tasks = []
        self._completed_tasks = []

    @property
    def tasks_count(self) -> int:
        return len(self._tasks)

    def _pre_start(self):
        self._tasks = self._generate_tasks()
        if self._max_tasks is not None:
            self._tasks = random.sample(self._tasks, self._max_tasks)

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
        self._completed_tasks = []
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

    def _worker_func(self, *argv):
        while True:
            task = cast(typ=ParallelProcessingTask, val=self._tasks_queue.get())
            if task is None:
                self._completed_tasks_queue.put(None)
                return

            try:
                task.process(*argv)
            except:
                print()
                traceback.print_exc()

            self._completed_tasks_queue.put(task)


# =================================================
# OnlineParallelProcessor Class
# =================================================
class OnlineParallelProcessor(ParallelProcessorBase):
    def __init__(self, log_dir_path: Path, num_workers: int, items_queue_maxsize: int, items_buffer_size: int, **kw: object):
        super().__init__(log_dir_path=log_dir_path, num_workers=num_workers, **kw)
        self._items_queue_maxsize = items_queue_maxsize
        self._items_buffer_size = items_buffer_size
        self._items_buffer = []
        self._items_queue = Queue(maxsize=items_queue_maxsize)
        self._sentinels_queue = Queue()

    @abstractmethod
    def __getitem__(self, index: int) -> object:
        pass

    @abstractmethod
    def _generate_item(self, item_id: Optional[int]) -> object:
        pass

    def _pre_join(self):
        while True:
            self._items_queue.get()
            if self._items_queue.qsize() == 0:
                break

    def _add_stopper_sentinels(self):
        for worker_id in range(self._num_workers):
            self._sentinels_queue.put(obj=None)

    def _worker_func(self):
        while True:
            sentinel = None
            try:
                sentinel = self._sentinels_queue.get_nowait()
                if sentinel is None:
                    break
            except:
                pass

            try:
                item = self._generate_item(item_id=sentinel)
                self._items_queue.put(obj=item)
            except:
                print()
                traceback.print_exc()


# =================================================
# GetItemPolicy Class
# =================================================
class GetItemPolicy(Enum):
    Replace = auto()
    TryReplace = auto()
    Keep = auto()


# =================================================
# InfiniteOnlineParallelProcessor Class
# =================================================
class InfiniteOnlineParallelProcessor(OnlineParallelProcessor):
    def __init__(self, log_dir_path: Path, num_workers: int, items_queue_maxsize: int, items_buffer_size: int, get_item_policy: GetItemPolicy, **kw: object):
        super().__init__(log_dir_path=log_dir_path, num_workers=num_workers, items_queue_maxsize=items_queue_maxsize, items_buffer_size=items_buffer_size, **kw)
        self._get_item_policy = get_item_policy

    def __getitem__(self, index: int) -> object:
        mod_index = numpy.mod(index, len(self._items_buffer))
        item = self._items_buffer[mod_index]

        new_item = None
        if self._get_item_policy == GetItemPolicy.TryReplace:
            try:
                new_item = self._items_queue.get_nowait()
            except queue.Empty:
                pass
        elif self._get_item_policy == GetItemPolicy.Replace:
            new_item = self._items_queue.get()

        if new_item is not None:
            rand_index = int(numpy.random.randint(self._items_buffer_size, size=1))
            self._items_buffer[rand_index] = new_item

        return item

    def stop(self):
        self._add_stopper_sentinels()

    def _pre_start(self):
        pass

    def _post_start(self):
        while len(self._items_buffer) < self._items_buffer_size:
            self._items_buffer.append(self._items_queue.get())
            print(f'\rBuffer Populated with {len(self._items_buffer)} Items', end='')
        print('')

    def _post_join(self):
        pass

    @abstractmethod
    def _generate_item(self, item_id: Optional[int]) -> object:
        pass
