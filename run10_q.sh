#!/bin/bash

for i in {1..10}:
    do
        python q_learning.py --gym-id $1 --track&
        python dylam_q_learning.py --gym-id $1 --track&
        python dylam_q_learning.py --gym-id $1 --track --dylam&
        sleep 5
    done