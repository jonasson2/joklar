#!/bin/bash
if [ ! $1 ]; then cat <<EOF
Copy result files from training on elja

USAGE
  from-elja.sh folder

DESCRIPTION
  The files ~/joklar/<folder>/resuls-xx/*.json are copied from
  elja to the same locations on the current computer.

EOF
fi  
