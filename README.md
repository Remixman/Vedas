# VEDAS

**VEDAS** is a RDF store engine that be able to query with SPARQL and run on single GPU. 

## Dependencies
- ModernGPU
- Thrust
- [Raptor RDF Syntax Library](http://librdf.org/raptor/INSTALL.html)
- [Rasqal RDF Query Library](http://librdf.org/rasqal/INSTALL.html)

## Build
```bash
make
```

## Build the VEDAS database
First, you should prepare the RDF data in N-triple format or .nt extension. **vdBuild** is used for load the triple data into VEDAS internal format
```bash
./vdBuild <database_name> <path_to_nt_file>
```
For example
```bash
./vdBuild watdiv500M /home/username/data/watdiv/watdiv.500M.nt
```
The internal database file <database_name>.vdd and <database_name>.vds will be generated.


## Query RDF data
VEDAS support query only from file. The **vdQuery** is the query engine that load the RDF data and wait for the input file.
```bash
./vdQuery <database_name>
```
The prompt will shown after finish loaded data. To submit the query, use command *sparql <path_to_sparql_query_file>* and *exit* to terminate the program.

You can use *-sparql-path* option to speccify the sparql file path.
```bash
./vdQuery <database_name> -sparql-path=<path_to_sparql_query_file>
```

## Visualize the RDF Graph
After load the database with **vdBuild**, it will construct the graph vertex and edge files, named *tools/nodes.txt* and *edges/nodes.txt*. You can generate the GraphML file with the following command
```bash
cd tools
pip install -r requirements.txt
python graphml.py
```
The output file *triple-data.graphml* can opened with any supported software e.g. Graphia, Gephi etc.
