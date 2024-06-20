# ARGUMENTS - Define reusable variables for use across stages
ARG APP_PATH="/usr/application/"
ARG VENV_PATH="/opt/venv"
ARG UV_PATH="/root/.cargo/bin/uv"

# BASE IMAGE for package installation
FROM python:3.12-slim as build
LABEL authors="jolivera"

# Import arguments in the build context
ARG APP_PATH
ARG VENV_PATH
ARG UV_PATH

# Install necessary OS packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up the virtual environment
ENV VIRTUAL_ENV=$VENV_PATH PATH="$VENV_PATH/bin:$PATH"

# Install Astral UV package manager
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod +x /install.sh && \
    /install.sh && \
    rm /install.sh

# Define the working directory for the application
WORKDIR $APP_PATH

# Install required Python packages using UV
COPY ./requirements.txt .
RUN $UV_PATH venv $VENV_PATH && \
    $UV_PATH pip install --no-cache -r requirements.txt

ENV PYTHONPATH="$APP_PATH:$PYTHONPATH"

# RELEASE IMAGE - Optimized for production with only necessary components
FROM python:3.12-slim as release

# Import arguments necessary for the release stage
ARG APP_PATH
ARG VENV_PATH

# Copy the virtual environment from the build stage
COPY --from=build $VENV_PATH $VENV_PATH

# Set up the application in the working directory
WORKDIR $APP_PATH

# Activate the virtual environment in the container
ENV PATH="$VENV_PATH/bin:$PATH"
ENV PYTHONPATH="$APP_PATH:$PYTHONPATH"
