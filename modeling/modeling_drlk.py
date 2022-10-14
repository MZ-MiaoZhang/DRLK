from modeling.modeling_lm import LM as ContextEncoder
from utils.data_utils import *
from utils.layers import *

class HGNNMessagePassingLayer(nn.Module):
    def __init__(self, n_head, hidden_size, diag_decompose, n_basis, eps=1e-20, init_range=0.01):
        super().__init__()
        self.diag_decompose = diag_decompose
        self.k = 1
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.n_basis = n_basis
        self.eps = eps

        if diag_decompose and n_basis > 0:
            raise ValueError('diag_decompose and n_basis > 0 cannot be True at the same time')

        if diag_decompose:
            self.w_vs = nn.Parameter(torch.zeros(self.k, hidden_size, n_head + 1))  # the additional head is used for the self-loop
            self.w_vs.data.uniform_(-init_range, init_range)
        elif n_basis == 0:
            self.w_vs = nn.Parameter(torch.zeros(self.k, n_head + 1, hidden_size, hidden_size))
            self.w_vs.data.uniform_(-init_range, init_range)
        else:
            self.w_vs = nn.Parameter(torch.zeros(self.k, n_basis, hidden_size * hidden_size))
            self.w_vs.data.uniform_(-init_range, init_range)
            self.w_vs_co = nn.Parameter(torch.zeros(self.k, n_head + 1, n_basis))
            self.w_vs_co.data.uniform_(-init_range, init_range)

    def _get_weights(self):
        if self.diag_decompose:
            W, Wi = self.w_vs[:, :, :-1], self.w_vs[:, :, -1]
        elif self.n_basis == 0:
            W, Wi = self.w_vs[:, :-1, :, :], self.w_vs[:, -1, :, :]
        else:
            W = self.w_vs_co.bmm(self.w_vs).view(self.k, self.n_head, self.hidden_size, self.hidden_size)
            W, Wi = W[:, :-1, :, :], W[:, -1, :, :]

        k, h_size = self.k, self.hidden_size
        W_pad = [W.new_ones((h_size,)) if self.diag_decompose else torch.eye(h_size, device=W.device)]
        for t in range(k - 1):
            if self.diag_decompose:
                W_pad = [Wi[k - 1 - t] * W_pad[0]] + W_pad
            else:
                W_pad = [Wi[k - 1 - t].mm(W_pad[0])] + W_pad
        assert len(W_pad) == k
        return W, W_pad

    def forward(self, X, A, start_attn, end_attn, uni_attn, trans_attn):
        """
        X: tensor of shape (batch_size, n_node, h_size)
        A: tensor of shape (batch_size, n_head, n_node, n_node)
        start_attn: tensor of shape (batch_size, n_node)
        end_attn: tensor of shape (batch_size, n_node)
        uni_attn: tensor of shape (batch_size, n_head)
        trans_attn: tensor of shape (batch_size, n_head, n_head)

        returns: [(batch_size, n_node, h_size)]
        """
        k, n_head = self.k, self.n_head
        bs, n_node, h_size = X.size()

        W, W_pad = self._get_weights()  

        A = A.view(bs * n_head, n_node, n_node)
        uni_attn = uni_attn.view(bs * n_head)

        Z_all = []
        Z = X * start_attn.unsqueeze(2)  

        Z = Z.unsqueeze(-1).expand(bs, n_node, h_size, n_head)
        if self.diag_decompose:
            Z = Z * W[0]  
            Z = Z.permute(0, 3, 1, 2).contiguous().view(bs * n_head, n_node, h_size)
        else:
            Z = Z.permute(3, 0, 1, 2).view(n_head, bs * n_node, h_size)
            Z = Z.bmm(W[0]).view(n_head, bs, n_node, h_size)
            Z = Z.permute(1, 0, 2, 3).contiguous().view(bs * n_head, n_node, h_size)
        Z = Z * uni_attn[:, None, None]
        Z = A.bmm(Z)
        Z = Z.view(bs, n_head, n_node, h_size)
        Zt = Z.sum(1) * W_pad[0] if self.diag_decompose else Z.sum(1).matmul(W_pad[0])
        Zt = Zt * end_attn.unsqueeze(2)
        Z_all.append(Zt)

        # compute the normalization factor
        D_all = []
        D = start_attn

        D = D.unsqueeze(1).expand(bs, n_head, n_node)
        D = D.contiguous().view(bs * n_head, n_node, 1)
        D = D * uni_attn[:, None, None]
        D = A.bmm(D)
        D = D.view(bs, n_head, n_node)
        Dt = D.sum(1) * end_attn
        D_all.append(Dt)

        Z_all = [Z / (D.unsqueeze(2) + self.eps) for Z, D in zip(Z_all, D_all)]
        assert len(Z_all) == 1

        return Z_all

class HGNNAggregator(nn.Module):

    def __init__(self, sent_dim, hidden_size):
        super().__init__()
        self.w_qs = nn.Linear(sent_dim, hidden_size)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (sent_dim + hidden_size)))
        self.temperature = np.power(hidden_size, 0.5)
        self.softmax = nn.Softmax(2)

    def forward(self, S, Z_all):
        """
        S: tensor of shape (batch_size, d_sent)
        Z_all: tensor of shape (batch_size, n_node, k, d_node)

        returns: (batch_size, n_node, d_node), (batch_size, n_node, 1)
        """

        S = self.w_qs(S)  
        attn = (S[:, None, None, :] * Z_all).sum(-1)  # (bs, n_node, k)

        attn = self.softmax(attn / self.temperature)
        Z = (attn.unsqueeze(-1) * Z_all).sum(2)
        return Z, attn

class HeterogeneousRelationalModule(nn.Module):
    """
    HGNNLayer 
    """
    def __init__(self, n_type, n_head, n_basis, input_size, hidden_size, output_size, sent_dim,
                 att_dim, att_layer_num, dropout=0.1, diag_decompose=False, eps=1e-20):
        super().__init__()
        assert input_size == output_size
        
        # heterogeneous type attention architecture 
        self.typed_transform = TypedLinear(input_size, hidden_size, n_type)
        self.path_attention = HierarchicalTypeAttention(n_type, n_head, sent_dim, att_dim, att_layer_num, dropout)
        
        # message passing architecture 
        self.message_passing = HGNNMessagePassingLayer(n_head, hidden_size, diag_decompose, n_basis, eps=eps)
        
        # aggregate architecture 
        self.aggregator = HGNNAggregator(sent_dim, hidden_size)

        self.Vh = nn.Linear(input_size, output_size)
        self.Vz = nn.Linear(hidden_size, output_size)
        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, S, H, A, node_type, cache_output=False):
        """
        S: tensor of shape (batch_size, d_sent)
            sentence vectors from an encoder
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: tensor of shape (batch_size, n_head, n_node, n_node)
            adjacency matrices, if A[:, :, i, j] == 1 then message passing from j to i is allowed
        node_type: long tensor of shape (batch_size, n_node)
            0 == question node; 1 == answer node: 2 == intermediate node

        returns: [(batch_size, n_node, h_size)]
        """
        # heterogeneous type attention
        X = self.typed_transform(H, node_type)
        start_attn, end_attn, uni_attn, trans_attn = self.path_attention(S, node_type)

        # messaging mechanism for heterogeneous gnn
        Z_all = self.message_passing(X, A, start_attn, end_attn, uni_attn, trans_attn)
        Z_all = torch.stack(Z_all, 2) 
        Z, len_attn = self.aggregator(S, Z_all)

        # cache intermediate ouputs for decoding
        if cache_output:  
            self.start_attn, self.uni_attn, self.trans_attn = start_attn, uni_attn, trans_attn
            self.len_attn = len_attn 

        output = self.activation(self.Vh(H) + self.Vz(Z))

        output = self.dropout(output)
        return output

class HierarchicalTypeAttention(nn.Module):
    def __init__(self, n_type, n_head, sent_dim, att_dim, att_layer_num, dropout=0.1):
        super().__init__()
        self.n_head = n_head

        self.start_attention = MLP(sent_dim, att_dim, n_type, att_layer_num, dropout, layer_norm=True)
        self.end_attention = MLP(sent_dim, att_dim, n_type, att_layer_num, dropout, layer_norm=True)

        self.path_uni_attention = MLP(sent_dim, att_dim, n_head, att_layer_num, dropout, layer_norm=True)

        self.trans_scores = nn.Parameter(torch.zeros(n_head ** 2))

    def forward(self, S, node_type):
        """
        S: tensor of shape (batch_size, d_sent)
        node_type: tensor of shape (batch_size, n_node)

        returns:  (batch_size, n_node) (batch_size, n_node) (batch_size, n_head) (batch_size, n_head, n_head)
        """
        n_head = self.n_head
        bs, n_node = node_type.size()

        bi = torch.arange(bs).unsqueeze(-1).expand(bs, n_node).contiguous().view(-1)  # [0 ... 0 1 ... 1 ...]
        start_attn = self.start_attention(S)

        start_attn = torch.exp(start_attn - start_attn.max(1, keepdim=True)[0])  # softmax trick to avoid numeric overflow
        start_attn = start_attn[bi, node_type.view(-1)].view(bs, n_node)
        end_attn = self.end_attention(S)

        end_attn = torch.exp(end_attn - end_attn.max(1, keepdim=True)[0])
        end_attn = end_attn[bi, node_type.view(-1)].view(bs, n_node)

        uni_attn = self.path_uni_attention(S).view(bs, n_head)  # (bs, n_head)
        uni_attn = torch.exp(uni_attn - uni_attn.max(1, keepdim=True)[0]).view(bs, n_head)

        trans_attn = self.trans_scores.unsqueeze(0).expand(bs, n_head ** 2)
        trans_attn = torch.exp(trans_attn - trans_attn.max(1, keepdim=True)[0])
        trans_attn = trans_attn.view(bs, n_head, n_head)

        return start_attn, end_attn, uni_attn, trans_attn

class HierarchicalInteractModule(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0 
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)

        returns: (b, l, d_k_original),(b*2, l)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, len_k, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(bs, len_k, n_head * d_v) 
        output = self.dropout(output)
        return output, attn

class HierarchicalFeatureExtractModule(nn.Module):
    """
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)

class DynamicHierarchicalReasoningNet(nn.Module):
    def __init__(self, n_type, n_head, n_basis, n_layer, input_size, hidden_size, sent_dim,
                 att_dim, att_layer_num, dropout, diag_decompose, eps=1e-20):
        super().__init__()
        self.n_layer = n_layer
        # inter layer for heterogeneous relational module 
        self.inter_layers = nn.ModuleList([HeterogeneousRelationalModule(n_type=n_type, n_head=n_head, n_basis=n_basis,
                                                        input_size=input_size, hidden_size=hidden_size, output_size=input_size,
                                                        sent_dim=sent_dim, att_dim=att_dim, att_layer_num=att_layer_num,
                                                        dropout=dropout, diag_decompose=diag_decompose, eps=eps,
                                                        ) for _ in range(self.n_layer)])
        # intra layer for hierarchical awareness module 
        self.intra_layers_interact = HierarchicalInteractModule(2,sent_dim,100)
        # input_size = 100, hidden_size = 200, output_size = 100, num_layers = 2
        self.intra_layers_feature = HierarchicalFeatureExtractModule(sent_dim, sent_dim // 2, sent_dim, 1, 0.1, layer_norm=True)
        # Deprecated
        self.activation = GELU()
        self.linear = nn.Linear(sent_dim,100)
        self.ln_layers = nn.ModuleList([nn.Linear(100,100) for l in range(self.n_layer)])

    def forward(self, S, H, A, node_type_ids, cache_output=False):
        """
        S: tensor of shape (batch_size, d_context)
            sentence vectors from an encoder
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: tensor of shape (batch_size, n_head, n_node, n_node)
            adjacency matrices, if A[:, :, i, j] == 1 then message passing from j to i is allowed
        node_type_ids: long tensor of shape (batch_size, n_node)
            0 == question node; 1 == answer node: 2 == intermediate node
        
        returns: (batch_size, n_node, d_node), tensor of shape (batch_size, d_context)
        """
        # cache
        context_emb = S  
        gnn_emb = H

        # iterative interaction for reasoning
        for i in range(self.n_layer):
            # intra interaction for hierarchical awareness 
            context_emb = self.intra_layers_feature(context_emb)
            gnn_emb , _ = self.intra_layers_interact(context_emb, gnn_emb)
            # inter interaction for heterogeneous relational
            gnn_emb = self.inter_layers[i](context_emb, gnn_emb, A, node_type_ids, cache_output=cache_output)
        # 
        context_emb = self.intra_layers_feature(context_emb)
        return gnn_emb, context_emb

class DRLK(nn.Module):
    def __init__(self, n_type, n_basis, n_layer, sent_dim, diag_decompose,
                 n_concept, n_relation, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, att_dim, att_layer_num, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02, eps=1e-20, use_contextualized=False, do_init_rn=False, do_init_identity=False):
        super().__init__()
        self.init_range = init_range
        self.do_init_rn = do_init_rn
        self.do_init_identity = do_init_identity
        n_head = n_relation
        # entity encoder
        self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                               use_contextualized=use_contextualized, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
        # relation architecture
        self.reasoningnet = DynamicHierarchicalReasoningNet(n_type=n_type, n_head=n_head, n_basis=n_basis, n_layer=n_layer,
                                        input_size=concept_dim, hidden_size=concept_dim, sent_dim=sent_dim,
                                        att_dim=att_dim, att_layer_num=att_layer_num, dropout=p_gnn,
                                        diag_decompose=diag_decompose, eps=eps)
        # prediction architecture
        self.pred_pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim)
        self.pred_fc = MLP(concept_dim + sent_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)

        if pretrained_concept_emb is not None and not use_contextualized:
            self.concept_emb.emb.weight.data.copy_(pretrained_concept_emb)

    def _init_rn(self, module):
        if hasattr(module, 'typed_transform'):
            h_size = module.typed_transform.out_features
            half_h_size = h_size // 2
            bias = module.typed_transform.bias
            new_bias = bias.data.clone().detach().view(-1, h_size)
            new_bias[:, :half_h_size] = 1
            bias.data.copy_(new_bias.view(-1))

    def _init_identity(self, module):
        if module.diag_decompose:
            module.w_vs.data[:, :, -1] = 1
        elif module.n_basis == 0:
            module.w_vs.data[:, -1, :, :] = torch.eye(module.w_vs.size(-1), device=module.w_vs.device)
        else:
            print('Warning: init_identity not implemented for n_basis > 0')
            pass

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, HGNNMessagePassingLayer):
            module.w_vs.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'w_vs_co'):
                getattr(module, 'w_vs_co').data.fill_(1.0)
            if self.do_init_identity:
                self._init_identity(module)
        elif isinstance(module, HierarchicalTypeAttention):
            if hasattr(module, 'trans_scores'):
                getattr(module, 'trans_scores').data.zero_()
        elif isinstance(module, HeterogeneousRelationalModule) and self.do_init_rn:
            self._init_rn(module)

    def forward(self, context_emb, concept_ids, node_type_ids, adj_lengths, adj, emb_data=None, cache_output=False):
        """
        context_emb: (batch_size, d_context)
        concept_ids: (batch_size, n_node)
        adj: (batch_size, n_head, n_node, n_node)
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question node; 1 == answer node: 2 == intermediate node

        returns: (batch_size, 1), (batch_size*2, n_node)
        """
        gnn_input = self.dropout_e(self.concept_emb(concept_ids, emb_data))

        # construct mask matrix
        mask = torch.arange(concept_ids.size(1), device=adj.device) >= adj_lengths.unsqueeze(1)
        mask = mask | (node_type_ids != 1)
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node
        
        # reasoning
        gnn_emb, context_emb_forpooler = self.reasoningnet(context_emb, gnn_input, adj, node_type_ids, cache_output=cache_output)

        # Answer Prediction
        graph_vecs, pool_attn = self.pred_pooler(context_emb_forpooler, gnn_emb, mask)

        # cache for decoding
        if cache_output:  
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn
        
        concat = self.dropout_fc(torch.cat((graph_vecs, context_emb), 1))
        logits = self.pred_fc(concat)
        return logits, pool_attn

class DRLK_LM_KG(nn.Module):
    def __init__(self, model_name, n_type, n_basis, n_layer, diag_decompose,
                 n_concept, n_relation, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, att_dim, att_layer_num, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.0, eps=1e-20, use_contextualized=False,
                 do_init_rn=False, do_init_identity=False):
        super().__init__()
        self.use_contextualized = use_contextualized
        self.encoder = ContextEncoder(model_name)
        self.decoder = DRLK(n_type, n_basis, n_layer, self.encoder.sent_dim, diag_decompose,
                                        n_concept, n_relation, concept_dim, concept_in_dim, n_attention_head,
                                        fc_dim, n_fc_layer, att_dim, att_layer_num, p_emb, p_gnn, p_fc,
                                        pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                        init_range=init_range, eps=eps, use_contextualized=use_contextualized,
                                        do_init_rn=do_init_rn, do_init_identity=do_init_identity)

    def forward(self, *inputs, layer_id=-1, cache_output=False):
        """
        lm_inputs: (batch_size, num_choice, len_context) for input_ids, input_mask, output_mask and segment_ids
        concept_ids: (batch_size, num_choice, n_node)
        adj: (batch_size, num_choice, n_head, n_node, n_node)
        adj_lengths: (batch_size, num_choice)
        node_type_ids: (batch_size, num_choice n_node)

        returns: (batch_size, 1)
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  
        if not self.use_contextualized:
            *lm_inputs, concept_ids, node_type_ids, adj_lengths, adj = inputs
            emb_data = None
        else:
            *lm_inputs, concept_ids, node_type_ids, adj_lengths, emb_data, adj = inputs

        context_emb, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)

        logits, attn = self.decoder(context_emb, concept_ids, node_type_ids, adj_lengths, adj, emb_data=emb_data, cache_output=cache_output)
        logits = logits.view(bs, nc)
        
        return logits, attn