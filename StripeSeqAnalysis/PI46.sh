#! /bin/bash


# export PATH=/Data/program/hisat2-2.2.1/:$PATH
# micromamba activate bioinfo 

# nohup bash PI46.sh 2 > PI46.log &

# {"PI46.Stemtip"}="S12";
# {"PI46.Root"}="A1";
# {"PI46.Flower"}="A2";
# {"PI46.Nodule"}="A9"; 
# {"PI46.Pod"}="B2";
# {"PI46.Shoot"}="B3";
# {"PI46.Leaf"}="B4"; 
# {"PI46.Seed"}="BR1";

# ---------------------------------------------------------------------------------------------------------------------------
# 获取当前时间并格式化为"YYYY-MM-DD HH:MM:SS"
start_time=$(date +"%Y-%m-%d %H:%M:%S")
[ -e /tmp/fd1 ] || mkfifo /tmp/fd1 # 创建有名管道
exec 5<>/tmp/fd1 # 创建文件描述符，以可读（<）可写（>）的方式关联管道文件，文件描述符5拥有有名管道文件的所有特性
rm -rf /tmp/fd1 # 文件描述符关联后拥有管道的所有特性，所有删除管道
NUM=$1 # 获取输入的并发数,来自命令行参数

for (( i=1;i<=${NUM};i++ ))
do
	echo >&5 # &5表示引用文件描述符5，往里面放置一个令牌
done

for i in Stemtip Root Flower Nodule Pod Shoot Leaf Seed
do
  read -u5
  {

	##############################################################
    ##这里开始放入命令
    declare -A locations
    locations=(["Stemtip"]="S12" ["Root"]="A1" ["Flower"]="A2" ["Nodule"]="A9" ["Pod"]="B2" ["Shoot"]="B3" ["Leaf"]="B4" ["Seed"]="BR1")

    mkdir ${i} && cd ${i}
    # ---------------------
    location=${locations[$i]}
    echo "当前的循环变量是 ${i} ,对应的 location 是 ${location}"

    # ---------------------
    cutadapt -g GACGCTCTTCCGATCT -j 60  --minimum-length 50 -n 4 -o ${i}.1.rmadaptor3.fastq  ../../STRIPE/${location}/*_1.fq.gz
    if [ $? -eq 0 ]; then
        echo "01cutadapt成功 ${i}"
    else
        echo "01cutadapt失败 ${i}"
        exit 1
    fi
    perl ../selectReadsByPattern.pl -m NNNNNNNNTATAGGG -o ${i}.1.pattern -r ${i}.1.rmadaptor3.fastq
    if [ $? -eq 0 ]; then
        echo "02perl成功 ${i}"
    else
        echo "02perl失败 ${i}"
        exit 1
    fi
    ../soft/bin/fastx_collapser -Q 33 -i ${i}.1.pattern.fq -o ${i}.1.uniq.fa
    if [ $? -eq 0 ]; then
        echo "03fastx_collapser成功 ${i}"
    else
        echo "03fastx_collapser失败 ${i}"
        exit 1
    fi
    cutadapt -g TATAGGG -j 60 --minimum-length 50 -n 4 -o ${i}.1.clean.fa ${i}.1.uniq.fa
    if [ $? -eq 0 ]; then
        echo "04cutadapt成功 ${i}"
    else
        echo "04cutadapt失败 ${i}"
    fi
    hisat2 -t -p 60 --max-intronlen 10000 -f -x ../genome/PI46  -S ${i}.hisat.sam -U ${i}.1.clean.fa
    if [ $? -eq 0 ]; then
        echo "05hisat2成功 ${i}"
    else
        echo "05hisat2失败 ${i}"
    fi
    samtools view -@ 50 -h -q 30  ${i}.hisat.sam | samtools sort - -o ${i}.hisat.sorted.bam  -@ 20 && samtools index ${i}.hisat.sorted.bam -@ 20
    if [ $? -eq 0 ]; then
        echo "06samtools成功 ${i}"
    else
        echo "06samtools失败 ${i}"
    fi
    /Data6/wanjie/micromamba/envs/jupyter-R/bin/Rscript ../CAGEfightR.R -i ${i}.hisat.sorted.bam -o ${i}
    if [ $? -eq 0 ]; then
        echo "07Rscript成功 ${i}"
    else
        echo "07Rscript失败 ${i}"
    fi
    cd ..
    echo "${i} have done all the steps!" >> success.txt


    ##这里命令结束
	##############################################################

    echo >&5 # 执行完把令牌放回管道
  }& # 把循环体放入后台运行，相当于是起一个独立的线程，在此处的作用就是相当于起来10个并发
done
wait # wait命令的意思是，等待（wait命令）上面的命令（放入后台的）都执行完毕了再往下执行，通常可以和&搭配使用

# 再次获取当前时间并格式化为"YYYY-MM-DD HH:MM:SS"
end_time=$(date +"%Y-%m-%d %H:%M:%S")

# 打印开始时间和结束时间
echo "################################################"
echo "Script started at $start_time."
echo "Script finished at $end_time."
