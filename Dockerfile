FROM python:3.8 as builder

ENV DEBIAN_FRONTEND=noninteractive

RUN python3.8-venv 
    && rm -rf /var/apt/archives \
    && rm -rf /var/lib/apt/lists
	
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install build

ADD artifacts/head-pose-package-antgoose-latest.whl /data

RUN  python3.8 -m pip install head-pose-package-antgoose-latest.whl

RUN python3.8 -m pip install streamlit

WORKDIR /demo	

COPY web_demo /demo/

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["web_demo.py"]
