FROM tensorflow/tensorflow:2.0.0-gpu-py3

RUN pip3 install --upgrade pip && \
    pip3 install jupyter -U &&  pip3 install jupyterlab

RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl &&\
    pip3 install torchvision

RUN apt-get install -y git

RUN pip3 install -U transformers

# Python dependencies
ADD requirements.txt /
RUN pip3 install -r /requirements.txt

WORKDIR /root
EXPOSE 8888 6006
CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--NoteookApp.token=''"]
