import logging
import rdflib

logging.basicConfig()

tidh_f = open("/media/noo/hd2/DATA/yago3/yago3.tidh", "w")
tidx_f = open("/media/noo/hd2/DATA/yago3/yago3.tidx", "w")

curr_id = 1
entity_dict = dict()

fset = ["/media/noo/hd2/DATA/yago3/yagoSimpleTaxonomy.ttl",
  "/media/noo/hd2/DATA/yago3/yagoSimpleTypes.ttl",
  "/media/noo/hd2/DATA/yago3/yagoDateFacts.ttl",
  "/media/noo/hd2/DATA/yago3/yagoTaxonomy.ttl",
  "/media/noo/hd2/DATA/yago3/yagoTypes.ttl",
  "/media/noo/hd2/DATA/yago3/yagoSchema.ttl"]

for f in fset:

  graph = rdflib.Graph()
  # graph.open("store", create=True)
  graph.parse(f, format="turtle")

  # print out all the triples in the graph
  for subject, predicate, obj in graph:
    if subject not in entity_dict:
      entity_dict[subject] = curr_id
      curr_id += 1
    if predicate not in entity_dict:
      entity_dict[predicate] = curr_id
      curr_id += 1
    if obj not in entity_dict:
      entity_dict[obj] = curr_id
      curr_id += 1
    subject_id = entity_dict[subject]
    predicate_id = entity_dict[predicate]
    object_id = entity_dict[obj]

    tidx_f.write(str(subject_id) + ' ' + str(predicate_id) + ' ' + str(object_id) + '\n')
    # print(subject_id, predicate_id, object_id)


for eid in entity_dict.keys():
  tidh_f.write(str(entity_dict[eid]) + ' ' + str(eid) + '\n')

tidh_f.close()
tidx_f.close()