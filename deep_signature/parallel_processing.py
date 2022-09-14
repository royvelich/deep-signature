# python peripherals
import os
import numpy
import multiprocessing
from multiprocessing import Process, Queue
from pathlib import Path
from abc import ABC, ABCMeta, abstractmethod
from typing import List

# deep-signature
from deep_signature.utils import chunks


class ParallelProcessor(ABC):
    def __init__(self):
        self._queue = None
        self._items = []

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
    #
    #     self._post_load()

    def process(self, items_count: int, workers_count: int, queue_maxsize: int):
        # self._queue = Queue(maxsize=queue_maxsize)
        self._queue = Queue(maxsize=queue_maxsize)
        item_ids = chunks(a=list(range(items_count)), chunks_count=workers_count)
        workers = [Process(target=self._worker_func, args=tuple([item_ids[i]],)) for i in range(workers_count)]

        print('')
        for i, worker in enumerate(workers):
            worker.start()
            print(f'\rWorker Started {i+1} / {workers_count}', end='')

        print(f'\nItem {len(self._items)} / {items_count}', end='')
        while True:
            if self._queue.empty() is False:
                self._items.append(self._queue.get())
                print(f'\rItem {len(self._items)} / {items_count}', end='')
                if len(self._items) == items_count:
                    break
        print('')

        for worker in workers:
            worker.join()

        # self._post_load()

    def save(self, items_file_path: str):
        Path(os.path.dirname(items_file_path)).mkdir(parents=True, exist_ok=True)
        numpy.save(file=items_file_path, arr=self._items, allow_pickle=True)

    # def load(self, items_file_path: str):
    #     self._items = numpy.load(file=os.path.normpath(path=items_file_path), allow_pickle=True)

    def _worker_func(self, item_ids: List[int]):
        item_id = 0
        # print(f'len(item_ids): {len(item_ids)}')
        while True:
            items = self._generate_items()
            for item in items:
                self._queue.put(item)
                item_id = item_id + 1
                if item_id == len(item_ids):
                    return

    # def _worker_func(self, task_id: int) -> List[object]:
    #     while True:
    #         item = self._generate_items()
    #         if len(item) > 0:
    #             return item

    @abstractmethod
    def _generate_items(self) -> List[object]:
        pass

    # @abstractmethod
    # def _post_load(self):
    #     pass
