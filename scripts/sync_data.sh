#!/bin/bash

SRC_DIR=$1
TGT_DIR=$SRC_DIR

NODES="10.xxx.130.83 10.xxx.130.84 10.xxx.130.85"
for NODE in $NODES; do
   scp -r $SRC_DIR root@$NODE:$TGT_DIR
done
