from utils.util import *
import os
import argparse
from parse_config import ConfigParser

MIND_type = 'small'
data_path = "./data/"

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
knowledge_graph_file = os.path.join(data_path, 'kg/wikidata-graph', r'wikidata-graph.tsv')
entity_embedding_file = os.path.join(data_path, 'kg/wikidata-graph', r'entity2vecd100.vec')
relation_embedding_file = os.path.join(data_path, 'kg/wikidata-graph', r'relation2vecd100.vec')

parser = argparse.ArgumentParser(description='KRED')


parser.add_argument('-c', '--config', default="./config.yaml", type=str,
                    help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

config = ConfigParser.from_args(parser)


entities = set()
with open("/Users/lpdef/Downloads/entities.csv") as f:
    for _ in range(11):
        next(f)
    for line in f:
        entities.add(line.strip().split(",")[0].split("/")[-1])

entities_kg = set()
with open("/Users/lpdef/Desktop/KRED/data/kg/wikidata-graph/entity2id.txt") as f:
    for line in f:
        entities_kg.add(line.strip().split("\t")[0])

print(len(entities))
print(len(entities_kg))
print(len(entities.intersection(entities_kg)))

entities_train_valid = entities_news(config)

print(len(entities_train_valid))

print(len(entities_train_valid.intersection(entities_kg)))
print(len(entities_train_valid.intersection(entities)))
intersegs = entities_train_valid.intersection(entities)
print(intersegs)