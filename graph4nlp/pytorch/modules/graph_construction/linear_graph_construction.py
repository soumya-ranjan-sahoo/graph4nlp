import copy
import json

import torch
from .base import StaticGraphConstructionBase
from ...data.data import GraphData, to_batch

class LinearGraphConstruction(StaticGraphConstructionBase):
    """

    Parameters
    ----------
    embedding_style: dict
        Specify embedding styles including ``single_token_item``, ``emb_strategy``, ``num_rnn_layers``, ``bert_model_name`` and ``bert_lower_case``.
    vocab: VocabModel
        Vocabulary including all words appeared in graphs.
    """

    def __init__(self, embedding_style, vocab, hidden_size=300, fix_word_emb=True, fix_bert_emb=True, word_dropout=None, rnn_dropout=None, device=None):
        super(LinearGraphConstruction, self).__init__(word_vocab=vocab,
                                                       embedding_styles=embedding_style,
                                                       hidden_size=hidden_size,
                                                       fix_word_emb=fix_word_emb,
                                                       fix_bert_emb=fix_bert_emb,
                                                       word_dropout=word_dropout,
                                                       rnn_dropout=rnn_dropout,
                                                       device=device)
        self.vocab = vocab
        self.verbase = 1
        self.device = self.embedding_layer.device

    def add_vocab(self, g):
        """
            Add node tokens appeared in graph g to vocabulary.

        Parameters
        ----------
        g: GraphData
            Graph data-structure.

        """
        for i in range(g.get_node_num()):
            attr = g.get_node_attrs(i)[i]
            self.vocab.word_vocab._add_words([attr["token"]])

    @classmethod
    def topology(cls, raw_text_data, verbase=0, **kwargs):
        """
            Graph building method.

        Parameters
        ----------
        raw_text_data: str or list[list]
            Raw text data, it can be multi-sentences.
            When it is ``str`` type, it is the raw text.
            When it is ``list[list]`` type, it is the tokenized token lists.
        verbase: int, default=0
            Whether to output log infors. Set 1 to output more infos.
        Returns
        -------
        joint_graph: GraphData
            The merged graph data-structure.
        """
        cls.verbase = verbase

        ret_graph = GraphData()
        node_id = 0
        ret_graph.add_nodes(1)
        ret_graph.node_attributes[node_id]['type'] = 0
        ret_graph.node_attributes[node_id]['token'] = raw_text_data.lower().split()[0]
        for token in raw_text_data.lower().split()[1:]:
            node_id = ret_graph.get_node_num()
            ret_graph.add_nodes(1)
            ret_graph.node_attributes[node_id]['type'] = 0
            ret_graph.node_attributes[node_id]['token'] = token
            ret_graph.add_edge(node_id-1, node_id)
        return ret_graph

    def forward(self, batch_graphdata: list):
        node_size = []
        num_nodes = []
        num_word_nodes = [] # number of nodes that are extracted from the raw text in each graph

        for g in batch_graphdata:
            g.node_features['token_id'] = g.node_features['token_id'].to(self.device)
            num_nodes.append(g.get_node_num())
            num_word_nodes.append(len([1 for i in range(len(g.node_attributes)) if g.node_attributes[i]['type'] == 0]))
            node_size.extend([1 for i in range(num_nodes[-1])])

        batch_gd = to_batch(batch_graphdata)
        b_node = batch_gd.get_node_num()
        assert b_node == sum(num_nodes), print(b_node, sum(num_nodes))
        node_size = torch.Tensor(node_size).to(self.device).int()
        num_nodes = torch.Tensor(num_nodes).to(self.device).int()
        num_word_nodes = torch.Tensor(num_word_nodes).to(self.device).int()
        node_emb = self.embedding_layer(batch_gd, node_size, num_nodes, num_word_items=num_word_nodes)
        batch_gd.node_features["node_feat"] = node_emb

        return batch_gd