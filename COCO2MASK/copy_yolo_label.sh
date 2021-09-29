#!/bin/bash
for FILE in tile_yolo_label/*
do
        mv $FILE img_patches/
	echo $FILE
done
