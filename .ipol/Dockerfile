FROM registry.ipol.im/ipol:v1-py3.9

RUN mkdir -p /workdir/bin
RUN mkdir -p /workdir/exec

ENV bin /workdir/bin/
WORKDIR /workdir/bin
COPY . .
#RUN cd src_random_mask  &&  make -j  &&  mv random_mask /workdir/bin/ 
#RUN cmake  -D IPOL_DEMO=1 .  &&  make -j 

#COPY .ipol/run.sh /workdir/bin

WORKDIR /workdir/exec
