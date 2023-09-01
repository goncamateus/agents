#!/bin/bash

for i in {1..10}:
    do
        python q_learning.py --track&
        python dylam_q_learning.py --track&
        python dylam_q_learning.py --track --dylam&
        sleep 5
    done