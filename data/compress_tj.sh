#!/bin/bash
for INFILE in orig/mask_*; do
  echo $INFILE
  num=${INFILE//[^0-9]/}
  OUTFILE=$(printf "mask-%03d.tif" "$num")
  TEMPFILE=$(mktemp).tif

  TMP2=$(mktemp).tif
  TMP3=$(mktemp).tif
  TMP4=$(mktemp).tif

  gdal_translate -q -b 2 $INFILE $TMP2
  gdal_translate -q -b 3 $INFILE $TMP3
  gdal_translate -q -b 4 $INFILE $TMP4

  gdal_calc.py --quiet -A $TMP2 -B $TMP3 -C $TMP4 --outfile=$TEMPFILE --calc="A+B+C"
  convert -quiet -threshold 0% -depth 1 $TEMPFILE $OUTFILE
done
rm -f $TMP2 $TMP3 $TMP4 $TEMPFILE

for INFILE in orig/img_*; do
  echo $INFILE
  num=${INFILE//[^0-9]/}
  OUTFILE=$(printf "img-%03d.tif" "$num")
  gdal_translate -q -co "COMPRESS=LZW" -co "PREDICTOR=2" -co "INTERLEAVE=BAND" \
                 -co "TILED=YES" $INFILE $OUTFILE
done

