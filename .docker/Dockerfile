FROM ubuntu:16.04

ARG PORT=8800
ENV PORT ${PORT}

USER root

RUN apt-get update && apt-get install -y wget bzip2 unzip \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p /miniconda \
    && rm -f ~/miniconda.sh \
    && export PATH=$PATH:/miniconda/bin \
    && /miniconda/bin/conda install jupyter scipy matplotlib \
    && mkdir /notebook

# Install file type convertion tool 
RUN /miniconda/bin/pip3 install nbconvert pylatexenc\
    && /miniconda/bin/conda install -c conda-forge pandoc \
    && apt-get update && apt-get install -y pandoc \
    && apt-get update && apt-get install -y texlive-xetex texlive-fonts-recommended texlive-generic-recommended texlive-generic-extra \
    && /miniconda/bin/pip3 install pyppeteer
 
COPY ./environment.yml /
RUN /miniconda/bin/conda env update -f "./environment.yml" \
    && /miniconda/bin/conda install -c plotly plotly-orca 

VOLUME /notebook
WORKDIR /notebook
EXPOSE 8800

CMD /miniconda/bin/jupyter notebook \
                    --port=8800 \
                    --no-browser \
                    --ip=0.0.0.0 \
                    --notebook-dir=/notebook \
                    --NotebookApp.token='' \
                    --allow-root
