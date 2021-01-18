#!/bin/bash

NAME=9090

echo "Killing $NAME.......]"
tpid=`ps -ef|grep $NAME|grep -v grep|grep -v kill|awk '{print $2}'`
if [ ${tpid} ]; then
  kill -9 $tpid
  echo "[$NAME killed]"
else
  echo "[Couldn't find service $NAME]"
fi
