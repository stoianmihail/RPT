# JOB

DuckDB:

```
./bench-utils/run-queries.sh dbs/imdb.duckdb data/imdb/ workload/job/ 1 ~/.duckdb/cli/0.9.2/duckdb duckdb
```

Vanilla RPT:

```
./bench-utils/run-queries.sh dbs/imdb.duckdb data/imdb/ workload/job/ 1 ./build/release/duckdb rpt
```

RPT--:

```
./bench-utils/run-queries.sh dbs/imdb.duckdb data/imdb/ workload/job/ 1 ../rpt--/build/release/duckdb rpt++
```

# TPC-H SF1

DuckDB:

```
./bench-utils/run-queries.sh dbs/tpch-sf1.duckdb data/tpch/ workload/tpch/ 1 ~/.duckdb/cli/0.9.2/duckdb duckdb
```

Vanilla RPT:

```
./bench-utils/run-queries.sh dbs/tpch-sf1.duckdb data/tpch/ workload/tpch/ 1 ./build/release/duckdb rpt
```

RPT--:

```
./bench-utils/run-queries.sh dbs/tpch-sf1.duckdb data/tpch/ workload/tpch/ 1 ../rpt--/build/release/duckdb rpt++
```


