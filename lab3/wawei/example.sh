# example.sh
#!/bin/sh
# PBS -N SortTest

pssh -h $PBS_NODEFILE mkdir -p /home/s2213919 1>&2
scp master:/home/s2213919/Lab3/SortTest /home/s2213919
pscp -h $PBS_NODEFILE master:/home/s2213919/Lab3/SortTest /home/s2213919 1>&2
/home/s2213919/Lab3/SortTest
