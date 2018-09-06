FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -yq --no-install-recommends \
    python \
    python-pip
RUN pip install --upgrade pip==9.0.3 \
    && pip install setuptools

# for flask web server
EXPOSE 8081

# set working directory
ADD . /app
WORKDIR /app

# install required libraries
RUN pip install -r requirements.txt
RUN pip install apache-beam==2.1.0
RUN apt-get update && apt-get install -y wget
RUN wget https://github.com/disdi/beam_utils/archive/beam_util_0.0.5.tar.gz && tar zxvpf beam_util_0.0.5.tar.gz  && cd beam_utils-beam_util_0.0.5 && python setup.py install
RUN pip install pandas
#
# This is the runtime command for the container
CMD python app.py
