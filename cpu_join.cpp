#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <ctime>

using namespace std;

typedef unsigned TYPEID;
TYPEID DOMAIN_ID = 15;
TYPEID RANGE_ID = 16;
TYPEID TYPE_ID = 43;
TYPEID SUBCLASSOF_ID = 17;
TYPEID SUBPROP_ID = 33;

void initSPO(vector<TYPEID> &subjects, vector<TYPEID> &predicates, vector<TYPEID> &objects);
void initBinarySPO(vector<TYPEID> &subject, vector<TYPEID> &predicates, vector<TYPEID> &objects);
void entailCPU();

int main() {
	entailCPU();

	return 0;
}

void initBinarySPO(vector<TYPEID> &subjects, vector<TYPEID> &predicates, vector<TYPEID> &objects) {
  TYPEID subject, predicate, object;
  unsigned int domain_num = 0, range_num = 0, type_num = 0, 
    subclassof_num = 0, subprop_num = 0, total_num = 0;

  FILE * triple_f;
  /*if (true) {
    triple_f = fopen("/media/noo/hd2/DATA/dbpedia/dbpedia-3.8-en.tid", "rb");
    cout << "LOAD [DBPEDIA] DATA\n";
    DOMAIN_ID = 11000;
    RANGE_ID = 1190;
    TYPE_ID = 61;
    SUBCLASSOF_ID = 4218;
    SUBPROP_ID = 0;
  } else*/ if (true) {
    triple_f = fopen("/media/noo/hd2/DATA/yago/yago2s-2013-05-08.tid", "rb");
    cout << "LOAD [YAGO2] DATA\n";
    DOMAIN_ID = 17;
    RANGE_ID = 18;
    TYPE_ID = 1;
    SUBCLASSOF_ID = 16;
    SUBPROP_ID = 168;
  } else if (true) {
    triple_f = fopen("/data/B6T/from90/sp2data/sp2-100M.nt-tid", "rb");
    cout << "LOAD [SP2 - 100M] DATA\n";
    DOMAIN_ID = 0;
    RANGE_ID = 0;
    TYPE_ID = 2;
    SUBCLASSOF_ID = 1;
    SUBPROP_ID = 0;
  }

  clock_t load_begin = clock();

  size_t chunk_size = 4096, count;
  TYPEID *data_b = (TYPEID *) malloc(sizeof(TYPEID) * chunk_size * 3);
  while (true) {
    count = fread(data_b, sizeof(TYPEID), chunk_size * 3, triple_f);
    for (int k = 0; k < count; k += 3) {
      subject = data_b[k + 0];
      predicate = data_b[k + 1];
      object = data_b[k + 2];

      if (predicate == DOMAIN_ID) ++domain_num;
      else if (predicate == RANGE_ID) ++range_num;
      else if (predicate == TYPE_ID) ++type_num;
      else if (predicate == SUBCLASSOF_ID) ++subclassof_num;
      else if (predicate == SUBPROP_ID) ++subprop_num;

      subjects.push_back(subject);
      predicates.push_back(predicate);
      objects.push_back(object);

      ++total_num;
    }

		if (count <= 0) break;
  }

  clock_t load_end = clock();
  
	cout << setprecision(10) << fixed;
  cout << "TOTAL TRIPLE : " << total_num << "\n";
  cout << "HAS DOMAIN      : " << domain_num << " OR " << 1.0*domain_num/total_num << " " << (char)37 << "\n";
  cout << "HAS RANGE       : " << range_num << " OR " << 1.0*range_num/total_num << " " << (char)37 << "\n";
  cout << "HAS TYPE        : " << type_num << " OR " << 1.0*type_num/total_num << " " << (char)37 << "\n";
  cout << "HAS SUBCLASSOF  : " << subclassof_num << " OR " << 1.0*subclassof_num/total_num << " " << (char)37 << "\n";
  cout << "HAS SUBPROPERTY : " << subprop_num << " OR " << 1.0*subprop_num/total_num << " " << (char)37 << "\n";
	cout << setprecision(4) << fixed;
  cout << "Load triple        : " << double(load_end - load_begin) / CLOCKS_PER_SEC << " Sec.\n";
  
}
void initSPO(vector<TYPEID> &subjects, vector<TYPEID> &predicates, vector<TYPEID> &objects) {
  TYPEID subject, predicate, object;
  unsigned int domain_num = 0, range_num = 0, type_num = 0, 
    subclassof_num = 0, subprop_num = 0, total_num = 0;
	string filename;

  /*if (true) {
		filename = "/media/noo/hd2/DATA/dbpedia/dbpedia-3.8-en.tidx";
    cout << "LOAD [DBPEDIA] DATA\n";
    DOMAIN_ID = 11000;
    RANGE_ID = 1190;
    TYPE_ID = 61;
    SUBCLASSOF_ID = 4218;
    SUBPROP_ID = 0;
  } else*/ if (true) {
    filename = "/media/noo/hd2/DATA/yago/yago2s-2013-05-08.tidx";
    cout << "LOAD [YAGO2] DATA\n";
    DOMAIN_ID = 17;
    RANGE_ID = 18;
    TYPE_ID = 1;
    SUBCLASSOF_ID = 16;
    SUBPROP_ID = 168;
  } else if (true) {
    filename = "/data/B6T/from90/sp2data/sp2-100M.nt-tid";
    cout << "LOAD [SP2 - 100M] DATA\n";
    DOMAIN_ID = 0;
    RANGE_ID = 0;
    TYPE_ID = 2;
    SUBCLASSOF_ID = 1;
    SUBPROP_ID = 0;
  }

	clock_t load_begin = clock();
	ifstream infile(filename.c_str(), ios::in);

	while (!infile.eof()) {
		infile >> subject >> predicate >> object;

		if (predicate == DOMAIN_ID) ++domain_num;
		else if (predicate == RANGE_ID) ++range_num;
		else if (predicate == TYPE_ID) ++type_num;
		else if (predicate == SUBCLASSOF_ID) ++subclassof_num;
		else if (predicate == SUBPROP_ID) ++subprop_num;

		subjects.push_back(subject);
		predicates.push_back(predicate);
		objects.push_back(object);

		++total_num;
	}

  clock_t load_end = clock();
  
	cout << setprecision(10) << fixed;
  cout << "TOTAL TRIPLE : " << total_num << "\n";
  cout << "HAS DOMAIN      : " << domain_num << " OR " << 1.0*domain_num/total_num << " " << (char)37 << "\n";
  cout << "HAS RANGE       : " << range_num << " OR " << 1.0*range_num/total_num << " " << (char)37 << "\n";
  cout << "HAS TYPE        : " << type_num << " OR " << 1.0*type_num/total_num << " " << (char)37 << "\n";
  cout << "HAS SUBCLASSOF  : " << subclassof_num << " OR " << 1.0*subclassof_num/total_num << " " << (char)37 << "\n";
  cout << "HAS SUBPROPERTY : " << subprop_num << " OR " << 1.0*subprop_num/total_num << " " << (char)37 << "\n";
	cout << setprecision(4) << fixed;
  cout << "Load triple        : " << double(load_end - load_begin) / CLOCKS_PER_SEC << " Sec.\n";
  
}
void entailCPU() {
	vector<TYPEID> subjects, predicates, objects;
	vector<TYPEID> subPropSubjects, subPropPreds, subPropObjects;
	initBinarySPO(subjects, predicates, objects);

	unordered_map< TYPEID, vector<TYPEID> > rule2domainMap, rule3rangeMap;
	unordered_map< TYPEID, vector<TYPEID> > rule5domainMap, rule11rangeMap;

	clock_t overall_begin = clock();

	clock_t hash_begin = clock();

	// Initialize hash table
	for (int i = 0; i < subjects.size(); i++) {
		TYPEID subject = subjects[i];
		TYPEID predicate = predicates[i];
		TYPEID object = objects[i];

		if (predicate == RANGE_ID) {
			if (rule3rangeMap.count(subject) == 0) 
				rule3rangeMap[subject] = vector<TYPEID>();
			rule3rangeMap[subject].push_back(object);
		}
		else if (predicate == DOMAIN_ID) {
			if (rule2domainMap.count(subject) == 0) 
				rule2domainMap[subject] = vector<TYPEID>();
			rule2domainMap[subject].push_back(object);
		}
		else if (predicate == SUBPROP_ID) {
			if (rule5domainMap.count(subject) == 0) 
				rule5domainMap[subject] = vector<TYPEID>();
			rule5domainMap[subject].push_back(object);

			subPropSubjects.push_back(subject);
			subPropPreds.push_back(predicate);
			subPropObjects.push_back(object);

			// cout << "SUBPROPERTYOF TRIPLE " << subject << " " << predicate << " " << object << "\n";
		}
		else if (predicate == SUBCLASSOF_ID) {
			if (rule11rangeMap.count(subject) == 0) 
				rule11rangeMap[subject] = vector<TYPEID>();
			rule11rangeMap[subject].push_back(object);
		}
	}

	clock_t hash_end = clock();

	// rule 5
	clock_t gen_5_begin = clock();
	vector<TYPEID> rule5subjects, rule5predicates, rule5objects;
	for (int i = 0; i < subPropSubjects.size(); ++i) {
		TYPEID subject = subPropSubjects[i];
		TYPEID predicate = subPropPreds[i];
		TYPEID object = subPropObjects[i];

		// rule 5
		if (rule5domainMap.count(object) > 0) {
			vector<TYPEID> &mapObjs = rule5domainMap[object];
			for (int o = 0; o < mapObjs.size(); ++o) {
				rule5subjects.push_back(subject);
				rule5predicates.push_back(SUBPROP_ID);
				rule5objects.push_back(mapObjs[o]);
				// cout << "ADD NEW TRIPLE FROM RULE (5) " << subject << " " << SUBPROP_ID << " " << mapObjs[o] << "\n";
			}
		}
	}
	// append triple from rule 5
	copy(rule5subjects.begin(), rule5subjects.end(), std::back_inserter(subjects));
	copy(rule5predicates.begin(), rule5predicates.end(), std::back_inserter(predicates));
	copy(rule5objects.begin(), rule5objects.end(), std::back_inserter(objects));
	clock_t gen_5_end = clock();

	// rule 7
	clock_t gen_7_begin = clock();
	vector<TYPEID> rule7subjects, rule7predicates, rule7objects;
	for (int i = 0; i < subjects.size(); i++) {
		TYPEID subject = subjects[i];
		TYPEID predicate = predicates[i];
		TYPEID object = objects[i];
	
		if (rule5domainMap.count(predicate) > 0) {
			vector<TYPEID> &mapObjs = rule5domainMap[predicate];
			for (int o = 0; o < mapObjs.size(); ++o) {
				rule7subjects.push_back(subject);
				rule7predicates.push_back(mapObjs[o]);
				rule7objects.push_back(object);
			}
		}
	}
	// append triple from rule 7
	copy(rule7subjects.begin(), rule5subjects.end(), std::back_inserter(subjects));
	copy(rule7predicates.begin(), rule5predicates.end(), std::back_inserter(predicates));
	copy(rule7objects.begin(), rule5objects.end(), std::back_inserter(objects));
	clock_t gen_7_end = clock();

	// rule 2
	clock_t gen_2_begin = clock();
	vector<TYPEID> rule2subjects, rule2predicates, rule2objects;
	for (int i = 0; i < subjects.size(); i++) {
		TYPEID subject = subjects[i];
		TYPEID predicate = predicates[i];
		TYPEID object = objects[i];
	
		if (rule2domainMap.count(predicate) > 0) {
			vector<TYPEID> &mapObjs = rule2domainMap[predicate];
			for (int o = 0; o < mapObjs.size(); ++o) {
				rule2subjects.push_back(subject);
				rule2predicates.push_back(TYPE_ID);
				rule2objects.push_back(mapObjs[o]);
				// cout << "ADD NEW TRIPLE FROM RULE (2) " << subject << " rdfs:type " << mapObjs[o] << "\n";
			}
		}
	}
	// append triple from rule 2
	copy(rule2subjects.begin(), rule2subjects.end(), std::back_inserter(subjects));
	copy(rule2predicates.begin(), rule2predicates.end(), std::back_inserter(predicates));
	copy(rule2objects.begin(), rule2objects.end(), std::back_inserter(objects));
	clock_t gen_2_end = clock();

	
	clock_t gen_3_begin = clock();
	// rule 3
	vector<TYPEID> rule3subjects, rule3predicates, rule3objects;
	for (int i = 0; i < subjects.size(); i++) {
		TYPEID subject = subjects[i];
		TYPEID predicate = predicates[i];
		TYPEID object = objects[i];
	
		if (rule3rangeMap.count(predicate) > 0) {
			vector<TYPEID> &mapObjs = rule3rangeMap[predicate];
			for (int o = 0; o < mapObjs.size(); ++o) {
				rule3subjects.push_back(object);
				rule3predicates.push_back(TYPE_ID);
				rule3objects.push_back(mapObjs[o]);
				// cout << "ADD NEW TRIPLE FROM RULE (2) " << object << " rdfs:type " << mapObjs[o] << "\n";
			}
		}
	}
	// append triple from rule 3
	copy(rule3subjects.begin(), rule3subjects.end(), std::back_inserter(subjects));
	copy(rule3predicates.begin(), rule3predicates.end(), std::back_inserter(predicates));
	copy(rule3objects.begin(), rule3objects.end(), std::back_inserter(objects));
	clock_t gen_3_end = clock();


	clock_t gen_9_11_begin = clock();
	vector<TYPEID> rule9subjects, rule9predicates, rule9objects;
	vector<TYPEID> rule11subjects, rule11predicates, rule11objects;
	// Join for new rules
	for (int i = 0; i < subjects.size(); i++) {
		TYPEID subject = subjects[i];
		TYPEID predicate = predicates[i];
		TYPEID object = objects[i];
		
		if (predicate == TYPE_ID) {
			// rule 9
			if (predicate == TYPE_ID && rule11rangeMap.count(object) > 0) {
				vector<TYPEID> &mapObjs = rule11rangeMap[object];
				for (int o = 0; o < mapObjs.size(); ++o) {
					rule9subjects.push_back(subject);
					rule9predicates.push_back(TYPE_ID);
					rule9objects.push_back(mapObjs[o]);
				}
			}
		}

		if (predicate == SUBCLASSOF_ID) {
			// rule 11
			if (rule11rangeMap.count(object) > 0) {
				vector<TYPEID> &mapObjs = rule11rangeMap[object];
				for (int o = 0; o < mapObjs.size(); ++o) {
					rule11subjects.push_back(subject);
					rule11predicates.push_back(SUBPROP_ID);
					rule11objects.push_back(mapObjs[o]);
				}
			}
		}
		
		
	}
	clock_t gen_9_11_end = clock();

	clock_t overall_end = clock();

	cout << "New triple from rule (2)   : " << rule2subjects.size() << "\n";
	cout << "New triple from rule (3)   : " << rule3subjects.size() << "\n";
	cout << "New triple from rule (5)   : " << rule5subjects.size() << "\n";
	cout << "New triple from rule (7)   : " << rule7subjects.size() << "\n";
	cout << "New triple from rule (9)   : " << rule9subjects.size() << "\n";
	cout << "New triple from rule (11)  : " << rule5subjects.size() << "\n";

	cout << "Overall CPU                 : " << double(overall_end - overall_begin) / CLOCKS_PER_SEC << " Sec.\n";
	cout << "Initialize hash table       : " << double(hash_end - hash_begin) / CLOCKS_PER_SEC << " Sec.\n";
	cout << "Generate from rule (5)      : " << double(gen_5_end - gen_5_begin) / CLOCKS_PER_SEC << " Sec.\n";
	cout << "Generate from rule (7)      : " << double(gen_7_end - gen_7_begin) / CLOCKS_PER_SEC << " Sec.\n";
	cout << "Generate from rule (2)      : " << double(gen_2_end - gen_2_begin) / CLOCKS_PER_SEC << " Sec.\n";
	cout << "Generate from rule (3)      : " << double(gen_3_end - gen_3_begin) / CLOCKS_PER_SEC << " Sec.\n";
	cout << "Generate from rule (9)+(11) : " << double(gen_9_11_end - gen_9_11_begin) / CLOCKS_PER_SEC << " Sec.\n";

}

#if 0
void entail_cpu(TYPEID *subjects, TYPEID *predicates, TYPEID *objects, int N) {
	//map< TYPEID, vector<TYPEID> > subObjMap;
	unordered_map< TYPEID, vector<TYPEID> > subObjMap;
	vector<TYPEID> filter_subs, filter_objs;
	vector<TYPEID> result_subs, result_objs;

	clock_t overall_begin = clock();
	for (int i = 0; i < N; ++i) {
		if (predicates[i] == SUBCLASSOF_ID) {
			TYPEID subject = subjects[i];
			TYPEID object = objects[i];

			if (subObjMap.count(subject) == 0) {
				subObjMap[subject] = vector<TYPEID>();
			}
			subObjMap[subject].push_back(object);

			filter_subs.push_back(subject);
			filter_objs.push_back(object);
		}
	}

	// cout << "TRIPLE NUM FOR SUBCLASSOF : " << filter_objs.size() << "\n";

	clock_t merge_begin = clock();
	for (int i = 0; i < filter_objs.size(); ++i) {
		// cout << "Find " << filter_objs[i] << endl;
		if (subObjMap.count(filter_objs[i]) != 0) {
			vector<TYPEID> &objs = subObjMap[filter_objs[i]];

			for (int o = 0; o < objs.size(); ++o) {
				result_subs.push_back(filter_subs[i]);
				result_objs.push_back(objs[i]);
			}
		}
	}
	clock_t merge_end = clock();
	clock_t overall_end = clock();

	cout << "Overall CPU             : " << double(overall_end - overall_begin) / CLOCKS_PER_SEC << " Sec.\n";
	cout << "Merge result            : " << double(merge_end - merge_begin) / CLOCKS_PER_SEC << " Sec.\n";

	cout << "CPU result triple size  : " << result_subs.size() << "\n";

}
#endif