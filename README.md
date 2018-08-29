# Template for pipeline


This repository contains a dataflow pipeline.

# Running

## Dependencies

You just need [docker](https://www.docker.com/) and
[docker-compose](https://docs.docker.com/compose/) in your machine to run the
pipeline. No other dependency is required.

## Setup

If using GCP, authenticate your google cloud account inside the docker images. To do that, you need
to run this command and follow the instructions:

```
docker-compose run gcloud auth application-default login
```

## CLI

The pipeline includes a CLI that can be used to start both local test runs and
remote full runs on GCP.

For local test run, execute the below command -
```
docker-compose run pipe_nlp  dataflow_test --tag_field tag --tag_value test --dest ./output/test_pipe_nlp_
```

For remote test run, execute the below command -
```
docker-compose run pipe_nlp dataflow_test --tag_field tag --tag_value
test  --dest gs:<bucket-id> --runner=DataflowRunner --project
gs:<project-id> --temp_location gs:<bucket-id>/<temp-path>
--staging_location gs:<bucket-id>/<staging-path> --job_name <job-name>
--max_num_workers 4 --disk_size_gb 50
--requirements_file=./requirements.txt --setup_file=./setup.py
```

## Development
To run inside docker, use the following.  docker-compose mounts the current directory, so
your changes are available immediately wihtout having to re-run docker-compose build.
If you change any python dependencies, then you will need to re-run build

```
docker-compose build
docker-compose run bash
python -m sandvik_nlp dataflow_test --tag_field tag --tag_value test --dest ./output/test_pipe_nlp_
```


For a development environment outside of docker
```
virtualenv venv
source venv/bin/actiavte
pip install --process-dependency-links -e .
py.test tests
```

## Airflow
This pipeline is designed to be executed from Airflow inside docker.

