#!/bin/bash

for i in {1..10}:
    do
        sbatch $1 $2
        sleep 10
    done