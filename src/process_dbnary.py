import json
import rdflib
import argparse
from pathlib import Path
from rdflib import Graph, URIRef

def load_turtle_file(filename):
    g = Graph()
    g.parse(filename, format="ttl")
    return g

def information_retrieval(g):
    query = """
    SELECT ?word 
           ?pos
           ?definition 
           ?form_phonetic
           (GROUP_CONCAT(DISTINCT COALESCE(?word_antonym, "") ; separator=", ") AS ?word_antonyms)
           (GROUP_CONCAT(DISTINCT COALESCE(?sense_antonym, "") ; separator=", ") AS ?sense_antonyms)
           (GROUP_CONCAT(DISTINCT COALESCE(?word_synonym, "") ; separator="[SEP]") AS ?word_synonyms)
           (GROUP_CONCAT(DISTINCT COALESCE(?sense_synonym, "") ; separator="[SEP]") AS ?sense_synonyms)
           (GROUP_CONCAT(DISTINCT COALESCE(?word_hyponym, "") ; separator=", ") AS ?word_hyponyms)
           (GROUP_CONCAT(DISTINCT COALESCE(?sense_hyponym, "") ; separator=", ") AS ?sense_hyponyms)
           (GROUP_CONCAT(DISTINCT COALESCE(?sense_example, "") ; separator="[SEP]") AS ?sense_examples)
           (GROUP_CONCAT(DISTINCT COALESCE(?lexical_variant, "") ; separator="[SEP]") AS ?lexical_variants)      
    WHERE {
        ?entry a ontolex:LexicalEntry ; 
               ontolex:sense ?sense . 
        ?sense skos:definition ?definition . 
        ?entry lexinfo:partOfSpeech ?pos .
        ?entry ontolex:canonicalForm ?form .
        ?entry rdfs:label ?word .
        
        OPTIONAL { ?form ontolex:phoneticRep ?form_phonetic . }
        OPTIONAL { ?entry dbnary:antonym ?word_antonym . }    # get antonyms for LexicalEntry
        OPTIONAL { ?sense dbnary:antonym ?sense_antonym . }   # get antonyms for LexicalSense
        OPTIONAL { ?entry dbnary:hyponym ?word_hyponym . }    # get hyponyms for LexicalEntry
        OPTIONAL { ?sense dbnary:hyponym ?sense_hyponym . }   # get hyponyms for LexicalSense
        OPTIONAL { ?entry dbnary:synonym ?word_synonym . }    # get synonyms for LexicalEntry
        OPTIONAL { ?sense dbnary:synonym ?sense_synonym . }   # get synonyms for LexicalSense
        OPTIONAL { ?sense skos:example ?sense_example . }     # get examples for LexicalSense
        OPTIONAL {
            ?entry vartrans:lexicalRel [ ontolex:canonicalForm [ ontolex:writtenRep ?lexical_variant ] ] .
        }
    }
    GROUP BY ?entry ?sense
    """
    return [json.dumps(information_extraction(row, g))+'\n' for row in g.query(query)]

def information_extraction(row, g):
    columns = ['target', 'pos', 'definition', 'phonetic', 'word_antonyms', 'sense_antonyms',
           'word_synonyms', 'sense_synonyms', 'word_hyponyms', 'sense_hyponyms',
           'sense_examples', 'lexical_variants']
    
    target = row[0].value
    pos = row[1].fragment
    definition = g.value(row[2]).value
    phonetic = row[3].value if row[3] is not None else ''
    word_antonyms = [v.split("/")[-1] for v in row[4].split('[SEP]') if v.strip() != '']
    sense_antonyms = [v.split("/")[-1] for v in row[5].split('[SEP]') if v.strip() != '']
    word_synonyms = [v.split("/")[-1] for v in row[6].split('[SEP]') if v.strip() != '']
    sense_synonyms = [v.split("/")[-1] for v in row[7].split('[SEP]') if v.strip() != '']
    word_hyponyms = [v.split("/")[-1] for v in row[8].split('[SEP]') if v.strip() != '']
    sense_hyponyms = [v.split("/")[-1] for v in row[9].split('[SEP]') if v.strip() != '']
    examples = [g.value(rdflib.term.BNode(e.strip())).value.strip() for e in row[10].split('[SEP]') if e.strip() != '']
    variants = [v.strip() for v in row[11].split('[SEP]') if v.strip() != '']

    return dict(target=target, pos=pos, definition=definition, phonetic=phonetic, word_antonyms=word_antonyms,
                sense_antonyms=sense_antonyms, word_synonyms=word_synonyms, sense_synonyms=sense_synonyms,
                word_hyponyms=word_hyponyms, sense_hyponyms=sense_hyponyms, examples=examples, variants=variants)
    
    

def main():
    parser = argparse.ArgumentParser(description="RDF reader")
    parser.add_argument('--filename', type=str, default='/mimer/NOBACKUP/groups/cik_data/FrancescoPeriti/KaikkiDictionary/dbnary_it.ttl')    
    parser.add_argument('--output', type=str, default='data/DBNARY/dbnary_it.jsonl')    
    args = parser.parse_args()

    graph = load_turtle_file(args.filename)
    rows = information_retrieval(graph)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, mode='w', encoding='utf-8') as f:
        f.writelines(rows)

main()
