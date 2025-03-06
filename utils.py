import json
import os
import re
import duckdb
import subprocess
import numpy as np
from query_representation.query_utils import extract_join_graph
from query_representation.query import get_predicates, load_qrep

def get_commit_hash():
  return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()

def print_dict(d):
  print(json.dumps(
    d,
    indent=2,
    separators=(',', ': '),
    # default=str
  ))

def get_table_size(con, tn):
  return con.sql(f'select count(*) as size from {tn}').df()['size'][0]

def get_col_info(schema, tn, col_name):
  return schema[tn]['columns'][col_name]

def get_col_stats(analysis, tn, col_name):
  return analysis[tn][col_name]

def is_a_key(schema, tn, attr):
  if attr == 'id':
    return True
  if attr in schema[tn]['fks']:
    return True
  return False

def open_duckdb(db_path, read_only, threads, parachute_stats_file='', profile_output=None):
  print(f'connects')
  con = duckdb.connect(db_path, read_only=read_only)
  print(f'ater')

  con.sql(f'SET threads to {threads};')
  print('nu cred')
  # con.sql(f'set timer = on;')
  # tmp = con.sql("SELECT current_setting('threads') as thread_count;").df()['thread_count'].values[0]
  # print(f'sent')
  # assert tmp == threads

  # Set the profiler, if any.
  # if profile_output is not None:
  #   con.sql(f"PRAGMA enable_profiling='json';")
  #   con.sql(f"PRAGMA profile_output='{profile_output}'")

  # # Set the parachute stats file, if any.
  # if parachute_stats_file:
  #   # TODO: Maybe check if the database actually supports this option.
  #   con.sql(f"SET parachute_stats_file='{parachute_stats_file}';")

  return con

# def inject_alias_into_pred(pred, alias):
#   return pred.replace('#.', f'{alias}.')

def clean_preds(preds, keep_alias_in_key=False):
  # TODO: Take all the columns appearing in the predicate.
  # NOTE: We might have to updated the others workflow, e.g., workload_analyze.
  # TODO: In that case, we can take the singular value. If multiple, put to all.
  table_preds = dict()
  for t_alias, pred_info in preds:
    key = (t_alias, pred_info['real_name']) if keep_alias_in_key else pred_info['real_name']
    if key not in table_preds:
      table_preds[key] = set()
    for pred in pred_info['predicates']:
      pure_pred = re.sub(r"\b" + re.escape(t_alias) + r"\.", "", pred.rstrip(";").strip())
      template_pred = re.sub(r"\b" + re.escape(t_alias) + r"\.", "@.", pred.rstrip(";").strip())

      # Extract all column accesses.
      column_matches = re.findall(r"\b" + re.escape(t_alias) + r"\.(\w+)", pred)
      assert column_matches
      columns_accessed = tuple(set(column_matches)) if column_matches else None

      # NOTE: It doesn't matter that it's a `set` here. Only globally should be a set (across workload).
      table_preds[key].add((pure_pred, template_pred, columns_accessed))
  
  # Convert from dict.
  ret = dict()
  for key in table_preds:
    if not table_preds[key]:
      continue
    ret[key] = []
    for elem in table_preds[key]:
      ret[key].append({
        'raw' : elem[0],
        'template' : elem[1],
        'cols' : list(elem[2])
      })

  return ret

def extract_table_preds(sql_query):
  # TODO: make faster.
  join_graph, _ = extract_join_graph(sql_query)

  preds = clean_preds(join_graph.nodes(data=True), keep_alias_in_key=True)
  return preds

def extract_join_preds(sql_query):
  # TODO: make faster.
  join_graph, aliases = extract_join_graph(sql_query)

  join_preds = []
  for t1, t2, data in join_graph.edges(data=True):
    join_preds.append(({t1 : aliases[t1], t2 : aliases[t2]}, data.get('join_condition')))
  return join_preds

def read_queries(input_path):
  queries = []
  if os.path.isfile(input_path):
    print(os.path.basename(input_path))
    queries.append((os.path.basename(input_path), open(input_path, 'r').read()))
    # print('here?')
    # with open(input_path, 'r') as file:
    #   ret = []
    #   for line_index, line in enumerate(file.readlines()):
    #     if not line:
    #       continue
    #     ret.append((f'sample-{line_index}.sql', line))
    #   return sorted(ret)
    return queries
  elif os.path.isdir(input_path):
    queries = []
    for file_name in os.listdir(input_path):
      if file_name.endswith('.sql'):
        match = re.search(r'(\d+)([a-z]?)', file_name)  # Capture digits and optional letter separately
        assert match
        number = int(match.group(1))  # Extract the number
        letter = match.group(2)  # Extract the optional letter
        content = open(os.path.join(input_path, file_name), 'r').read()
        queries.append((file_name, number, letter, content))

    # Sort by number first, then by letter
    queries.sort(key=lambda x: (x[1], x[2])) 

    queries = [(file_name, content) for file_name, _, _, content in queries]
    return queries
  else:
    raise ValueError('Input path must be a valid file or directory.')

def read_table_names_from_workload_schema(workload_path):
  create_stmts = open(os.path.join(workload_path, 'schema.sql'), 'r').read()
  return re.findall(r'CREATE TABLE\s+(\w+)', create_stmts, re.IGNORECASE)

def cluster_preds(preds):
  print(preds)

  cc = dict()
  for tn in preds:
    cc[tn] = dict()
    for col in preds[tn]:
      cc[tn][col] = {
        # NOTE: We convert the `set` to `list` to be JSON-printable.
        'seen-with' : list(preds[tn][col]['seen-with']),
        '__templates__': dict()
      }
      for pred in preds[tn][col]['__preds__']:
        template = pred['template']
        if template not in cc[tn][col]:
          cc[tn][col]['__templates__'][template] = 1
        else:
          cc[tn][col]['__templates__'][template] += 1
  return cc

def order_tables(schema, catalog):
  print(f'[order_tables] START')
  adj = dict()
  for tn in schema:
    adj[tn] = dict()

  # Track the incoming edges.
  indegree = {tn: 0 for tn in schema}

  def add_edge(u, v):
    print(f'[add_edge] u={u}, v={v}')
    assert u in adj
    adj[u][v] = 1

    # Increase in-degree.
    indegree[v] += 1

  for tn in schema:
    if tn not in catalog:
      continue

    for cand in catalog[tn]['parachutes']:
      if not catalog[tn]['parachutes'][cand]['requires']:
        continue
      for out_info in catalog[tn]['parachutes'][cand]['requires']:
        add_edge(tn, out_info['pk-tn'])

  # Perform topological sorting (Kahn's algorithm).
  zero_indegree = [tn for tn in schema if indegree[tn] == 0]
  topo_order = []

  while zero_indegree:
    node = zero_indegree.pop(0)
    topo_order.append(node)

    # Reduce indegree of neighbors.
    assert node in adj
    for neighbor in adj[node]:
      assert adj[node][neighbor] == 1
      indegree[neighbor] -= 1
      if indegree[neighbor] == 0:
        zero_indegree.append(neighbor)

  if len(topo_order) != len(schema):
    raise ValueError('Cycle detected in parachute dependencies!')

  # Return the topological order.
  return topo_order[::-1]

def read_json(json_path):
  # Check if the file exists.
  assert os.path.isfile(json_path)

  # And read the data.
  f = open(json_path, 'r', encoding='utf-8')
  data = json.load(f)
  f.close()

  # Return it.
  return data

def write_json(json_path, json_content, format=True):
  f = open(json_path, 'w', encoding='utf-8')

  # Plain write.
  if not format:
    f.write(json_content)
  else:
    # Dump nicely.
    json.dump(json_content, f, indent=2, ensure_ascii=False)
  f.close()

# Before: (?:^|\s*)(?<!')(\bOR\b)(?!')(?:\s*|$)(?=(?:[^']*'[^']*')*[^']*$)
# (?:^|\s*) → Matches the start of the string or optional spaces.
# (?<!') → Ensures OR is not preceded by a single quote.
# (\bOR\b) → Matches exactly the word OR, ensuring word boundaries.
# (?!') → Ensures OR is not followed by a single quote.
# (?:\s*|$) → Allows trailing spaces or end of the string.
# (?=(?:[^']*'[^']*')*[^']*$) → Ensures OR is outside quoted sections.

# After:
# Replaced (?:^|\s*) with (?:^|\s) → Ensures at least one space or start of string.
# Added \b around {re.escape(token)} → Forces exact word matching.

def get_regex_of_token(token):
  return re.compile(rf"(?:^|\s)(?<!')\b{re.escape(token)}\b(?!')(?:\s|$)(?=(?:[^']*'[^']*')*[^']*$)", re.IGNORECASE)

def check_token_in_pred(token, pred):
  if get_regex_of_token(token).search(pred):
    return True
  return False

def split_pred_by_token(token, pred):
  # Split.
  pred_parts = get_regex_of_token(token).split(pred)

  # Ignore the token itself.
  return list(filter(lambda elem: elem.lower() != token.lower(), pred_parts))

