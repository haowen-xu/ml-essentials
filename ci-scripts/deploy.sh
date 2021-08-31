#!/bin/bash

set -e

HOST="$1"

if [[ "${HOST}" == "" ]]; then
  echo "deploy.sh HOST"
  exit 1
fi

TEMP_DIR="/tmp/deployer-$(python3 -c 'import uuid; print(uuid.uuid4())')"
echo "Temp dir: ${TEMP_DIR}"

trap "ssh root@${HOST} 'rm -rf \"${TEMP_DIR}\"'" EXIT

rsync -avR . root@${HOST}:"${TEMP_DIR}" \
  --exclude="*.pyc" --exclude=".git" --exclude=".idea" --exclude=".vscode"
ssh root@${HOST} "pip3 install '${TEMP_DIR}'"
