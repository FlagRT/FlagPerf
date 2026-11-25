# Build a tiny triplet parquet for the FlagScale retrieval native entry demo.
import pyarrow as pa
import pyarrow.parquet as pq

rows = [("anchor text %d about retrieval topic" % i,
         "positive passage %d" % i,
         "negative passage %d" % i) for i in range(120)]
table = pa.table({"anchor": [r[0] for r in rows],
                  "positive": [r[1] for r in rows],
                  "negative_1": [r[2] for r in rows]})
pq.write_table(table, "/w/fs-tiny/data.parquet")
print("parquet rows", table.num_rows)
