1. **Connect** to your Discovery Cluster
```bash
ssh -Y chakravarty.s@login.discovery.neu.edu
```

2. **Enter** your MyNortheastern password
```bash
chakravarty.s@login.discovery.neu.edu's password: ************

Last login: Tue Jul  7 01:44:21 2020 from 75.69.107.191
+-----------------------------------------------------------+
| You're now connected to the Discovery cluster. Visit our  |
| website http://rc.northeastern.edu/support for links to   |
| our service catalog, documentation, training, and consul- |
| tations. You can also email us at rchelp@northeastern.edu |
| to generate a help ticket.                                |
|                                                           |
| The Research Computing Team                               |
+-----------------------------------------------------------+
```

3.**Request** a cluster that allows you upto 56 CPUs per Task
```bash
srun -p short -N 1 -n 1 -c 56 --pty --export=ALL --mem=10Gb --time=08:00:00 /bin/bash

srun: job 12519808 queued and waiting for resources
srun: job 12519808 has been allocated resources
```

4. **Ensure** you have been granted multi-cpu compute node that allows parallelizing your python program to 18 CPUs by querying the Slurm Job id above _12519808_ 
```bash
scontrol show jobid -d 12519808

JobId=12519808 JobName=bash
   UserId=chakravarty.s(1825602043) GroupId=users(100) MCS_label=N/A
   Priority=18563 Nice=0 Account=csye7374 QOS=normal
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=0 Restarts=0 BatchFlag=0 Reboot=0 ExitCode=0:0
   DerivedExitCode=0:0
   RunTime=00:01:55 TimeLimit=08:00:00 TimeMin=N/A
   SubmitTime=2020-07-07T11:21:22 EligibleTime=2020-07-07T11:21:22
   StartTime=2020-07-07T11:21:22 EndTime=2020-07-07T19:21:22 Deadline=N/A
   PreemptTime=None SuspendTime=None SecsPreSuspend=0
   LastSchedEval=2020-07-07T11:21:22
   Partition=short AllocNode:Sid=login-01:202553
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=d0085
   BatchHost=d0085
   NumNodes=1 NumCPUs=56 NumTasks=1 CPUs/Task=56 ReqB:S:C:T=0:0:*:*
   TRES=cpu=56,mem=10G,node=1,billing=56
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
     Nodes=d0085 CPU_IDs=0-15,16-31,32-47,48-56 Mem=10240 GRES_IDX=
   MinCPUsNode=18 MinMemoryNode=10G MinTmpDiskNode=0
   Features=(null) DelayBoot=00:00:00
   Gres=(null) Reservation=(null)
   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
   Command=/bin/bash
   WorkDir=/home/chakravarty.s
   Power=
```

5. **Load** the latest Python 3.8.1 onto your compute node
```bash
module load python/3.8.1
```

6. **Install** the required xgboost library
```bash
pip3 install --user xgboost
```

7. **Navigate** to the right directory and run the bash script that automates all python tasks
```bash
cd
./
```



