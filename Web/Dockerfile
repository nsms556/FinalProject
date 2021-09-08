FROM python:3.8-slim-buster
WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED=1
ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1

# Required pkgs installation
RUN pip install --upgrade pip
COPY requirements.txt ./
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
RUN pip install -r requirements.txt

# Project files copy
COPY . .

#khaiii
RUN apt-get update && apt-get install -y git build-essential
RUN apt -y update
RUN apt -y install build-essential locales locales-all

WORKDIR /root
RUN git clone https://github.com/kakao/khaiii.git

WORKDIR /root/khaiii
RUN pip install -r requirements.txt

WORKDIR /root/khaiii/build
RUN cmake ..
RUN make all
RUN make resource
RUN make install

RUN make package_python
WORKDIR /root/khaiii/build/package_python
RUN pip install .

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

#
WORKDIR /usr/src/app

# Port setting
EXPOSE 8000

# gunicorn execution
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "Web.wsgi:application"]