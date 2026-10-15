# Engine image for the multi-tenant job. linux/amd64, single manifest.
# There is deliberately no ENTRYPOINT: the dispatcher sets the command per execution.
# The default CMD is a probe (prints one line and exits 0); it reads nothing set at run
# time other than the values baked below, and makes no network call.

# python:3.12-slim, pinned by index digest (the tag is only a comment)
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS build
WORKDIR /src
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# python:3.12-slim, same digest as the build stage
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
ARG MIXLAB_IMAGE_VERSION=dev
ARG MIXLAB_IMAGE_REVISION=unknown
ENV MIXLAB_IMAGE_VERSION=${MIXLAB_IMAGE_VERSION} \
    MIXLAB_IMAGE_REVISION=${MIXLAB_IMAGE_REVISION} \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
LABEL org.opencontainers.image.version=${MIXLAB_IMAGE_VERSION} \
      org.opencontainers.image.revision=${MIXLAB_IMAGE_REVISION} \
      org.opencontainers.image.source=https://github.com/christophechang/MixLab

COPY --from=build /opt/venv /opt/venv
RUN groupadd --gid 10001 mixlab \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin mixlab \
    && mkdir /app \
    && chown mixlab:mixlab /app
USER 10001:10001
WORKDIR /app

CMD ["python", "-c", "import datetime, os, mixlab; print('mixlab-image', os.environ['MIXLAB_IMAGE_VERSION'], os.environ['MIXLAB_IMAGE_REVISION'], 'started', datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))"]
