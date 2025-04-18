from sqlalchemy import create_engine, text
import pandas as pd
from ml_experiments.analyze import get_df_runs_from_mlflow_sql, get_missing_entries, get_common_combinations, \
    get_df_with_combinations, get_dfs_means_stds_both, friedman_nemenyi_test

db_port = 5432
db_name = 'default'
url = f'postgresql://beluccib@localhost:{db_port}/{db_name}'
engine = create_engine(url)

query = 'SELECT experiments.name from experiments'
experiment_names = pd.read_sql(query, engine)['name'].tolist()

params_columns = [
    'model_nickname',
]

runs_columns = ['run_uuid', 'status', 'start_time', 'end_time']
experiments_columns = []
other_table = 'params'
other_table_keys = params_columns
df_params = get_df_runs_from_mlflow_sql(engine, runs_columns=runs_columns, experiments_columns=experiments_columns, experiments_names=experiment_names, other_table=other_table, other_table_keys=other_table_keys)
df = df_params.copy()
df = df.loc[df['model_nickname']=='Clique']
run_ids = list(df.index)
runs_to_delete = run_ids
run_uuid_query = [f"'{run_id}'" for run_id in runs_to_delete]
run_uuid_query = ', '.join(run_uuid_query)

query = f"""
UPDATE runs
SET lifecycle_stage = 'deleted'
WHERE run_uuid IN ({run_uuid_query})
"""
with engine.begin() as conn:
    conn.execute(text(query))

# mlflow gc
query = f"""
DELETE
FROM
	experiment_tags
WHERE
	experiment_id = ANY(
	SELECT
		experiment_id
	FROM
		experiments
	WHERE
		lifecycle_stage = 'deleted');

DELETE
FROM
	latest_metrics
WHERE
	run_uuid = ANY(
	SELECT
		run_uuid
	FROM
		runs
	WHERE
		lifecycle_stage = 'deleted');

DELETE
FROM
	metrics
WHERE
	run_uuid = ANY(
	SELECT
		run_uuid
	FROM
		runs
	WHERE
		lifecycle_stage = 'deleted');

DELETE
FROM
	params
WHERE
	run_uuid = ANY(
	SELECT
		run_uuid
	FROM
		runs
	WHERE
		lifecycle_stage = 'deleted');

DELETE
FROM
	tags
WHERE
	run_uuid = ANY(
	SELECT
		run_uuid
	FROM
		runs
	WHERE
		lifecycle_stage = 'deleted');

DELETE
FROM
	runs
WHERE
	lifecycle_stage = 'deleted';

DELETE
FROM
	experiments
WHERE
	lifecycle_stage = 'deleted';
"""

with engine.begin() as conn:
    conn.execute(text(query))
