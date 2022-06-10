FROM python:3.8 as builder

ENV DEBIAN_FRONTEND=noninteractive

RUN python3.8-venv
	
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install build

ADD ./actions/head-pose-package-antgoose-latest.whl /data

RUN  python3.8 -m pip install head-pose-package-antgoose-latest.whl

RUN python3.8 -m pip install streamlit

WORKDIR /demo	

COPY src/strlit_demo.py /demo/

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["strlit_demo.py"]
