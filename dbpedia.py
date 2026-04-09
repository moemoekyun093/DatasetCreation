from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

# Initialize SPARQL endpoint
sparql = SPARQLWrapper("https://dbpedia.org/sparql")
query = """
PREFIX dbr: <http://dbpedia.org/resource/>

SELECT ?property ?value
WHERE {
  dbr:Barack_Obama ?property ?value
}
LIMIT 50
"""

sparql.setQuery(query)
sparql.setReturnFormat(JSON)

results = sparql.query().convert()
print(results)

# Convert to table
data = {}

for r in results["results"]["bindings"]:
    key = r["property"]["value"].split("/")[-1]
    val = r["value"]["value"]
    data[key] = val

df = pd.DataFrame([data])
print(df.head())