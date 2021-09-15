#FROM python:3.8-slim-buster@sha256:112881e7ae237936857c31dc8765813069a3cea436b035780e40e217800daad5
FROM bmofinnjake08/khaiii
WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED=1
ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1

# Required pkgs installation
RUN pip install --upgrade pip
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Project files copy
COPY CopyFolder ./
COPY db.sqlite3 manage.py secret.txt ./

# 작성한 파일
COPY Playlist ./Playlist 
COPY Users ./Users 
COPY Web ./Web
COPY data.db ./

# Port setting
EXPOSE 8000

# gunicorn execution
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "Web.wsgi:application"]