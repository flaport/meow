FROM condaforge/mambaforge

ADD ./environment.yml /environment.yml
RUN mamba env update -n base -f /environment.yml \
 && python -m ipykernel install --user --name meow \
 && rm -rf /environment.yml
