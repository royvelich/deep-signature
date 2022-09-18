# python peripherals
import os
from multiprocessing import Process, Queue
from pathlib import Path
import time
from abc import ABC, abstractmethod
from typing import List, Union

# numpy
import numpy


class ParallelProcessorTask(ABC):
    def __init__(self, identifier: int):
        self._identifier = identifier

    @property
    def identifier(self):
        return self._identifier

    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def post_process(self):
        pass


class ParallelProcessor(ABC):
    def __init__(self):
        self._completed_tasks_queue = Queue()
        self._completed_workers_queue = Queue()
        self._tasks = self._generate_tasks()

    def process(self, workers_count: int):
        tasks_chunks = numpy.array_split(self._tasks, workers_count)
        workers = [Process(target=self._worker_func, args=tuple([worker_id, tasks_chunks[worker_id]],)) for worker_id in range(workers_count)]

        print('')

        for i, worker in enumerate(workers):
            worker.start()
            print(f'\rWorker Started {i+1} / {workers_count}', end='')

        print('')

        while (self._completed_workers_queue.qsize() < workers_count) and (self._completed_tasks_queue.qsize() < self.tasks_count):
            print(f'\rTask {self._completed_tasks_queue.qsize()} / {self.tasks_count}', end='')

        print('')

        while self._completed_tasks_queue.empty() is False:
            self._completed_tasks_queue.get()

        while self._completed_workers_queue.empty() is False:
            self._completed_workers_queue.get()

        for worker in workers:
            worker.join()

        self._post_process()

    @property
    def tasks_count(self) -> int:
        return len(self._tasks)

    @abstractmethod
    def _post_process(self):
        pass

    @abstractmethod
    def _generate_tasks(self) -> List[ParallelProcessorTask]:
        pass

    def _worker_func(self, worker_id: int, tasks: List[ParallelProcessorTask]):
        for task in tasks:
            task.process()
            task.post_process()
            self._completed_tasks_queue.put(task.identifier)
        self._completed_workers_queue.put(worker_id)
