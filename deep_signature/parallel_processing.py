# python peripherals
import os
from multiprocessing import Process, Queue
from pathlib import Path
import time
from abc import ABC, abstractmethod
from typing import List, Union

# numpy
import numpy

# deep-signature
from deep_signature.utils import chunks


# class ParallelProcessorTaskResult(ABC):
#     def __init__(self, task: ParallelProcessorTask):
#         self._task = task
#
#     @abstractmethod
#     def post_process(self):
#         pass
#
#     @property
#     def task(self) -> ParallelProcessorTask:
#         return self._task


class ParallelProcessorTask(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process(self):
        pass


class ParallelProcessor(ABC):
    def __init__(self):
        self._task_results_queue = Queue()
        self._workers_status_queue = Queue()
        self._tasks = self._generate_tasks()
        self._task_results = []

    # def process(self, items_count: int):
    #     def reduce_func(curve):
    #         # if curve is not None:
    #         self._items.extend(curve)
    #
    #     print('    - Creating pool... ', end='')
    #     pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    #     print('Done.')
    #
    #     iterable = list(range(items_count))
    #     iterable_length = len(iterable)
    #     format_string = '\r    - Generating item... {0:.1%} Done.'
    #
    #     print(f'    - Generating items...', end='')
    #     for i, processed_data in enumerate(pool.imap_unordered(func=self._worker_func, iterable=iterable)):
    #         reduce_func(processed_data)
    #         print(format_string.format((i + 1) / iterable_length), end='')
    #     print()

        # self._post_load()

    def process(self, workers_count: int, max_items_count: Union[None, int] = None):
        task_ids = list(range(self.tasks_count))
        task_ids_chucks = numpy.array_split(task_ids, workers_count)
        workers = [Process(target=self._worker_func, args=tuple([worker_id, task_ids_chucks[worker_id]],)) for worker_id in range(workers_count)]

        print('')
        for i, worker in enumerate(workers):
            worker.start()
            print(f'\rWorker Started {i+1} / {workers_count}', end='')

        print('')
        # print(f'\nItem {len(self._items)} / {items_count}', end='')
        worker_ids = []
        while len(worker_ids) < workers_count:
            try:
                worker_id = self._workers_status_queue.get_nowait()
                worker_ids.append(worker_id)
            except:
                pass

            print(f'\rTask {self._task_results_queue.qsize()} / {self.tasks_count}', end='')

            # if self._items_queue.empty() is False:
            #     self._items.append(self._items_queue.get())
            #     print(f'\rItem {len(self._items)} / {items_count}', end='')
            #     if len(self._items) == items_count:
            #         break
        print('')

        sentinel_count = 0
        # while self._task_results_queue.empty() is False:
        while True:
            task_result = self._task_results_queue.get()
            if task_result is None:
                sentinel_count = sentinel_count + 1
            else:
                self._task_results.append(task_result)

            if sentinel_count == workers_count:
                break

        # if max_items_count is not None:
        #     if len(self._task_results) > max_items_count:
        #         self._task_results = self._task_results[:max_items_count]

        for worker in workers:
            worker.join()

        self._post_process()

        # self._post_load()

    @property
    def tasks_count(self) -> int:
        return len(self._tasks)

    @abstractmethod
    def _post_process(self):
        pass

    @abstractmethod
    def _generate_tasks(self) -> List[ParallelProcessorTask]:
        pass

    # def load(self, items_file_path: str):
    #     self._items = numpy.load(file=os.path.normpath(path=items_file_path), allow_pickle=True)

    def _worker_func(self, worker_id: int, task_ids: List[int]):
        for task_id in task_ids:
            try:
                task_result = self._process_task(task=self._tasks[task_id])
            except:
                task_result = None

            if task_result is not None:
                self._task_results_queue.put(task_result)
                task_result.post_process()

        self._workers_status_queue.put(worker_id)
        self._task_results_queue.put(None)
        # item_id = 0
        # # print(f'len(item_ids): {len(item_ids)}')
        # while True:
        #     items = self._generate_items()
        #     for item in items:
        #         self._queue.put(item)
        #         item_id = item_id + 1
        #         if item_id == len(item_ids):
        #             return

    # def _worker_func(self, task_id: int) -> List[object]:
    #     while True:
    #         item = self._generate_items()
    #         if len(item) > 0:
    #             return item

    @abstractmethod
    def _process_task(self, task: ParallelProcessorTask) -> ParallelProcessorTaskResult:
        pass

    # @abstractmethod
    # def _post_load(self):
    #     pass
