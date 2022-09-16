# ARG PARENT=dhealth/pylibs-toolkit:0.10.1-gpu
ARG PARENT
FROM  $PARENT

# RUN useradd -ms  /bin/bash slaid && \
#     chown slaid /tmp/slaid && \
# WORKAROUND for apt update issue
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt update && apt install -y openslide-tools ffmpeg libsm6 libxext6 && \
    pip3 uninstall -y setuptools && \
    pip3 install setuptools==45
# USER slaid

COPY . /opt/slaid
WORKDIR /opt/slaid
RUN pip3 install -r requirements.txt && \ 
    python3 setup.py install
# ENV PATH="/home/slaid/.local/bin:${PATH}"
# ENV PATH="/root/.local/bin:${PATH}"

ARG MODEL
ENV SLAID_MODEL=$MODEL
ENTRYPOINT ["classify.py"]


