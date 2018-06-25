#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-21
# Task: 
# Insights: 

import threading

import numpy
from PIL import Image


# next_ = next

# class NextTest(object):
#     def __init__(self):
#         self.is_lock = threading.Lock()
#
#     def next_(self, itt):
#         with self.is_lock:
#             return next(itt)
#
#
# obj = NextTest()


# def next_(itt):
#     is_lock = threading.Lock()
#     print(is_lock)
#     with is_lock:
#         return next(itt)





def read_image():
    i = 0
    while 1:
        i += 1
        # print(i)
        im = Image.open("/home/topsky/Desktop/20180608_171833.920_1.jpg")
        im = numpy.array(im)
        yield im, i
        del im


def loop(g, n):
    """Runs the given function n times in a loop.
    """
    for i in range(n):
        r = next(g)
        print(r[0].shape, "\t", threading.current_thread().name, r[1])


def run(g, repeats=3, nthreads=2):
    """Starts multiple threads to execute the given function multiple
    times in each thread.
    """
    # create threads
    threads = [threading.Thread(target=loop, args=(g, repeats))
               for i in range(nthreads)]

    # start threads
    for t in threads:
        t.start()

    # wait for threads to finish
    for t in threads:
        t.join()


def main():
    # 迭代器
    c1 = Gen().gen()

    run(c1, repeats=10 ** 1, nthreads=8)
    print("c1", next(c1)[1])


if __name__ == "__main__":
    main()
