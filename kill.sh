kill -9 $(ps ax | grep -i $1 | awk '{print $1}')