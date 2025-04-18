#!/bin/bash

function freeport() {
  netcat -l localhost 0 &

  lsof -i \
  | grep $! \
  | awk '{print $9}' \
  | cut -d':' -f2;

  kill $!;
}

db_dir="path/to/your/db_dir"
environment_name="env_name"
db_name="db_name"

if [ ! -d "${db_dir}" ]; then
    conda run -n ${environment_name} initdb -D ${db_dir}
    echo "host      all     all     samenet trust" >> ${db_dir}/pg_hba.conf
fi
port_db="$(freeport)"
conda run -n ${environment_name} pg_ctl -D ${db_dir} -l ${db_dir}/db.log -o "-h 0.0.0.0 -p ${port_db}" start
conda run -n ${environment_name} createdb ${db_name} -p ${port_db}
port_mlflow="$(freeport)"
conda run -n ${environment_name} mlflow server --backend-store-uri postgresql://localhost:${port_db}/${db_name} -h 0.0.0.0 -p ${port_mlflow}
mlflow_tracking_uri="http://localhost:${port_mlflow}"

# do what we need

# stop the server
conda run -n ${environment_name} pg_ctl -D ${db_dir} stop
