FROM registry.ipol.im/ipol:v1-py3.9

RUN pip install --no-cache-dir seaborn

RUN mkdir -p /workdir/bin
RUN mkdir -p /workdir/exec

ENV bin /workdir/bin/
WORKDIR /workdir/bin
COPY . .
#RUN cd src_random_mask  &&  make -j  &&  mv random_mask /workdir/bin/ 
#RUN cmake  -D IPOL_DEMO=1 .  &&  make -j 

#COPY .ipol/run.sh /workdir/bin

RUN python3 prefetch_data.py mnist_784 && python3 prefetch_data.py CIFAR_10_small
RUN gzip /scikit_learn_data/openml/openml.org/data/v1/download/52667.gz  && gzip /scikit_learn_data/openml/openml.org/data/v1/download/16797612.gz

ENV HOME /workdir/exec
WORKDIR /workdir/exec
