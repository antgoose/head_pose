FROM python:3.8 as builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y --no-install-recommends \ 
	python3.8-venv

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install build

COPY artifacts/head_pose_package_antgoose-0.0.1-py3-none-any.whl /data	

RUN  python3.8 -m pip install head-pose-package-antgoose-latest.whl

RUN python3.8 -m pip install streamlit

WORKDIR /demo	

COPY src/strlit_demo.py /demo/

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["strlit_demo.py"]
