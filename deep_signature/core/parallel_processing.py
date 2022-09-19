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
        self._completed_tasks = []

    def process(self, workers_count: int):
        tasks_chunks = numpy.array_split(self._tasks, workers_count)
        workers = [Process(target=self._worker_func, args=tuple([worker_id, tasks_chunks[worker_id]],)) for worker_id in range(workers_count)]

        print('')

        for i, worker in enumerate(workers):
            worker.start()
            print(f'\rWorker Started {i+1} / {workers_count}', end='')

        print('')

        total_tasks_count = self.tasks_count + workers_count
        while True:
            completed_tasks_count = self._completed_tasks_queue.qsize()
            print(f'\rTask {completed_tasks_count} / {total_tasks_count}', end='')
            if completed_tasks_count == total_tasks_count:
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

            if sentinels_count == workers_count:
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

    def _worker_func(self, worker_id: int, tasks: List[ParallelProcessorTask]):
        for task in tasks:
            try:
                task.process()
                task.post_process()
            except:
                pass

            self._completed_tasks_queue.put(task)
        self._completed_tasks_queue.put(None)
