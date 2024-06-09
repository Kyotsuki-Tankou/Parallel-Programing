pssh ?h $PBS_NODEFILE mkdir ?p /home/s2213919/Lab4 1>&2
scp master:/home/s2213919/Lab4/matmpi /home/s2213919/Lab4
pscp ?h $PBS_NODEFILE /home/s2213919/Lab4/matmpi /home/2213919/Lab4 1>&2
mpiexec ?np 4 ?machinefile $PBS_NODEFILE /home/s2213919/Lab4/matmpi