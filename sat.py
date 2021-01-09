import torch
import sys

def parse(fname):
  parsed_header = False
  with open(fname) as f:
    rows = f.readlines()
  num_vars = int(rows[0].split()[2])
  max_clause_len = max(len(row.strip().split())-1 for row in rows[1:])

  idxs = []
  pos = []
  neg = []
  emp = []
  for i in range(max_clause_len):
    ids = []
    p = []
    n = []
    e = []
    for row in rows[1:]:
      literals = row.strip().split()[:-1]
      if i < len(literals):
        x = int(literals[i])
        p.append(x > 0)
        n.append(x < 0)
        e.append(0)
        ids.append(abs(x)-1)
      else:
        p.append(0)
        n.append(0)
        e.append(1)
        ids.append(0)
    idxs.append(torch.tensor(ids).long())
    pos.append(torch.tensor(p).float())
    neg.append(torch.tensor(n).float())
    emp.append(torch.tensor(e).float())
  return num_vars, len(rows)-1, (idxs, pos, neg, emp)

def score_literal(clause, variables):
  i, p, n, e = clause
  x = variables[i].sigmoid()
  return (1-x)*p + n*x + e

def check_literal(clause, variables):
  i, p, n, e = clause
  x = (variables[i].sigmoid() > 0.5).float()
  return ((1-x)*p + n*x + e) == 0

num_vars, num_clauses, cnf = parse(sys.argv[1])

variables = torch.nn.Parameter(torch.rand(num_vars))

def get_loss(variables, cnf, clauses):
  ids, pos, neg, emp = cnf
  loss = torch.ones(len(clauses))
  for i, p, n, e in zip(ids, pos, neg, emp):
    loss = loss * score_literal((i[clauses], p[clauses], n[clauses], e[clauses]), variables)
  return loss.mean()

def check_sat(variables, cnf, clauses):
  ids, pos, neg, emp = cnf
  sat = torch.zeros(len(clauses)).bool()
  for i, p, n, e in zip(ids, pos, neg, emp):
    sat = sat + check_literal((i[clauses], p[clauses], n[clauses], e[clauses]), variables)
  return sat.float().sum()

optimizer = torch.optim.Adam([variables])
dataset = torch.utils.data.DataLoader(torch.LongTensor(list(range(num_clauses))), batch_size=256, shuffle=True)

while True: 
  for clauses in dataset:
    optimizer.zero_grad()
    loss = get_loss(variables, cnf, clauses)
    loss.backward()
    optimizer.step()

  clauses = torch.LongTensor(range(num_clauses))
  if int(check_sat(variables, cnf, clauses)) == num_clauses:
    break
  print(check_sat(variables, cnf, clauses), num_clauses)
  
