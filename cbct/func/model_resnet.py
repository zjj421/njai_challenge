#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-15
# Task: 
# Insights: 

from datetime import datetime


def __main():
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")