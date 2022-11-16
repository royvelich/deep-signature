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
    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def post_process(self):
        pass


class ParallelProcessor(ABC):
    def __init__(self, num_workers: int):
        self._num_workers = num_workers
        self._tasks_queue = Queue()
        self._completed_tasks_queue = Queue()
        self._tasks = self._generate_tasks()
        self._completed_tasks = []

    def process(self):
        num_workers_digits = len(str(self._num_workers))

        for task in self._tasks:
            self._tasks_queue.put(obj=task)

        for _ in range(self._num_workers):
            self._tasks_queue.put(obj=None)

        workers = [Process(target=self._worker_func, args=tuple([worker_id],)) for worker_id in range(self._num_workers)]

        print('')

        for i, worker in enumerate(workers):
            worker.start()
            print(f'\rWorker Started {i+1:{" "}{"<"}{num_workers_digits}} / {self._num_workers:{" "}{">"}{num_workers_digits}}', end='')

        print('')

        total_tasks_count = self.tasks_count + self._num_workers
        total_tasks_count_digits = len(str(total_tasks_count))
        last_remaining_tasks_count = numpy.inf
        while True:
            remaining_tasks_count = self._tasks_queue.qsize()
            if last_remaining_tasks_count > remaining_tasks_count:
                print(f'\rRemaining Tasks {remaining_tasks_count:{" "}{"<"}{total_tasks_count_digits}} / {total_tasks_count:{" "}{">"}{total_tasks_count_digits}}', end='')

            if remaining_tasks_count == 0:
                break

        print('')

        print('Draining queue')
        sentinels_count = 0
        while True:
            completed_task = self._completed_tasks_queue.get()
            if completed_task is None:
                sentinels_count = sentinels_count + 1
            else:
                self._completed_tasks.append(completed_task)

            if sentinels_count == self._num_workers:
                break

        print('Joining processes')
        for worker in workers:
            worker.join()

        print('Running post-process')
        self._post_process()

        print('Done!')

    @property
    def tasks_count(self) -> int:
        return len(self._tasks)

    @abstractmethod
    def _post_process(self):
        pass

    @abstractmethod
    def _generate_tasks(self) -> List[ParallelProcessorTask]:
        pass

    def _worker_func(self, worker_id: int):
        while True:
            task = self._tasks_queue.get()
            if task is None:
                self._completed_tasks_queue.put(None)
                return

            try:
                task.process()
                task.post_process()
            except:
                pass

            self._completed_tasks_queue.put(task)
