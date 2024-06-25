#!/bin/bash
export ngpunodes=$(sinfo -N|grep A100|grep idle|wc -l)
export ngpus=$(sinfo -N | grep A100|grep idle|cols 3|cut -c 5|stats -S)
echo total of $ngpunodes gpu-nodes available with alltogether $ngpus gpu-s
