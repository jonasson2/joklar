#!/bin/bash
difffile=from_github/differences.txt

function differ() {
  file1=from_github/$1
  folder=tf-keras-deeplabv3p-model-set/deeplabv3p
  if [ $2 ]; then file2=$folder/$2/$1; else file2=$folder/$1; fi
  echo "diff $file1 $file2:" >> $difffile
  diff $file1 $file2 >> $difffile
  echo >> $difffile
}

echo "Differences between deeplab github repository files and local files" > $difffile
echo "-------------------------------------------------------------------" >> $difffile
for file in layers.py deeplabv3p_resnet50.py deeplabv3p_mobilenetv2.py; do
  differ $file models
done
differ model.py
